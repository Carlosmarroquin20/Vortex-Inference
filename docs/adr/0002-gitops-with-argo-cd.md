# 0002. GitOps with Argo CD: staging and prod promoted by commit

- Status: accepted
- Date: 2026-11-09

## Context

Before this change, `make deploy` ran `kubectl apply -k` from a laptop. The
cluster's state was whatever the last person applied, and nothing put back
a manual change. There was also no record of which image was running where,
other than the cluster itself. ADR 0001 gives every commit on main a signed
image. What was missing was a way to get that image to running
environments, in order, without a person and without credentials into the
cluster.

The requirements were:

- two environments, staging and prod, in separate namespaces, with prod
  more conservative;
- each merge to main reaches staging, and reaches prod only if staging is
  healthy, with no human approval;
- secrets live in git like everything else, but readable only by the
  cluster;
- manual changes in the cluster are reverted;
- the existing local workflow (`make setup`, `make deploy`) keeps working
  unchanged.

## Decision

**Argo CD with an app of apps.** `make bootstrap-gitops`
(`scripts/bootstrap-gitops.sh`) is the only imperative step. It copies the
model onto the node and installs the Sealed Secrets key and Argo CD. It
then applies one root Application. The root syncs `gitops/apps/`, which
declares one Application per component in sync waves:

| Wave | Applications |
|------|--------------|
| -3 | Argo CD, which then manages itself |
| -2 | Sealed Secrets, KEDA, Argo Rollouts, kube-prometheus-stack |
| -1 | cluster-scoped shared objects |
| 0 | staging and prod |

Every Application has automated sync with `prune` and `selfHeal`. Every
upstream manifest is referenced by commit SHA, and every image by digest. A
CI check fails on any image without a digest.

**Environments are Kustomize overlays.** `k8s/overlays/staging` and
`k8s/overlays/prod` build on `k8s/base`, the same base `make deploy` uses,
through a shared component (`k8s/components/gitops-environment`).

- **The component** removes `spec.replicas`, because KEDA owns the count.
  It moves cluster-scoped objects to `gitops/cluster`, since one object
  cannot belong to two Applications. It also sets the image pull policy and
  the model.
- **Staging** runs one to two replicas with the base disruption budget,
  `maxUnavailable: 1`.
- **Prod** runs three to four replicas with `minAvailable: 2`.
- **Monitoring** rules no longer name a namespace. Prometheus runs with
  `enforcedNamespaceLabel`, so each environment's copy of the rules only
  sees, and only alerts on, its own namespace.

**Promotion is a commit.** After `release.yml` verifies an image, the
`staging` job runs `kustomize edit set image` on the staging overlay with
`tag@digest` and pushes the commit to main (`scripts/promote.sh`). A
cluster bootstrapped from that commit must sync it, report every
Application Healthy, and pass `scripts/smoke-test.sh`. The smoke test
checks that every pod runs the promoted digest, then sends an authenticated
completion and an unauthenticated one, which must be rejected. If it
passes, the `production` job commits the same `tag@digest` to the prod
overlay and runs the same checks. It then runs `scripts/drift-check.sh`,
which changes prod's replica floor by hand and requires Argo CD to restore
it. The git history of `k8s/overlays/*/kustomization.yaml` is the
deployment log, and reverting a promotion commit is the rollback.

**Ephemeral verification clusters.** No cluster runs permanently behind
this repository. Each gate creates a kind cluster on the runner and
bootstraps it with `make bootstrap-gitops` at the promotion commit. It
waits for Argo CD, not kubectl, to bring the environment up, and tears it
down at the end. Pointed at a long-lived cluster, the same gates would wait
on that Argo CD's Application status instead. The bootstrap code,
manifests and checks would not change.

**Secrets with Sealed Secrets.** Each environment's API key and RabbitMQ and
Redis credentials are committed as SealedSecrets, encrypted with
`--scope strict` semantics. Each value only decrypts into its own Secret
name and namespace. The sealing certificate is in the repository. The
private key is in `.secrets/` (gitignored) and in the `SEALED_SECRETS_KEY`
repository secret. The bootstrap installs it before the controller starts,
so every cluster built from this repository can open the same secrets.
`scripts/seal_secrets.py` generates and seals fresh credentials.

## Alternatives considered

- **Flux.** It covers the same ground, and its image-automation controllers
  would do the tag update in-cluster. Argo CD was chosen for its UI and
  application health model, and because Argo Rollouts (ADR 0003) is from
  the same project and integrates with its health checks and resource
  actions.
- **One ApplicationSet instead of an app of apps.** A list or git-directory
  generator would stamp out staging and prod from one template. The
  platform components are not uniform, though: Helm and Kustomize, different
  namespaces and sync options. Sync waves between generated and hand-written
  Applications also need the same health customization anyway. Explicit
  Applications are more readable for eight objects.
- **Argo CD Image Updater.** It writes new tags back to git itself. But it
  promotes whatever appears in the registry, with no gate between staging
  and prod. The gate is the point of this design, and CI already knows
  which digest it verified.
- **Promotion by branch (a `staging` and a `prod` branch) or by a separate
  config repository.** Environment branches drift apart and make "what
  differs between staging and prod" a merge problem. A config repository
  is the better choice when application and platform teams differ, but here
  it would split one change across two repositories. One main branch with
  one directory per environment keeps the diff between environments
  visible in a single file.
- **SOPS (with age or KMS) or External Secrets Operator.** SOPS needs the
  decryption key in Argo CD, through a plugin or KSOPS, and the key ends up
  in the same place as the sealing key. External Secrets needs an external
  secret store (Vault, a cloud secret manager) and an account to hold it.
  Sealed Secrets needs neither and keeps the ciphertext reviewable in pull
  requests.
- **A manual approval before prod (GitHub environment protection).** It is
  common, but requirement 2 rules out a human gate. Here staging health,
  and from ADR 0003 the canary analysis, is the gate.
- **A long-lived cluster for staging and prod.** This would be the
  production shape, but it needs a cloud account and money. The ephemeral
  clusters exercise the same GitOps path on every promotion, at the cost
  of checking an environment as it starts rather than as it has been
  running.

## Consequences

- Every merge to main produces two follow-up commits from
  `github-actions[bot]`: `deploy(staging): sha-…` and `deploy(prod): sha-…`.
  Commits pushed with the workflow token do not trigger workflows, so they
  cannot loop.
- A manual change to a managed object is reverted within seconds. The only
  intentional exceptions are the replica counts, which KEDA owns.
- The environments serve Qwen2.5-0.5B, the model the evaluation gate
  scores, sized so staging and prod fit on one node next to the platform.
  Moving them to Phi-3 is a change to the component's ConfigMap patch and
  resources, on a cluster with more capacity.
- Losing the sealing key means resealing every secret
  (`scripts/seal_secrets.py`). The key in `.secrets/` must be backed up.
- `make bootstrap-gitops` on minikube needs about 6 CPUs and 10 GB. Both
  environments and the platform share the node.
