# 0004. End-to-end test on an ephemeral kind cluster, through git

- Status: accepted
- Date: 2026-11-12

## Context

ADRs 0001–0003 make several claims, and each can break without any unit
test noticing:

- Argo CD can build the whole platform from git.
- The SealedSecrets open.
- Manual changes are reverted.
- A good canary is promoted.
- A bad one is aborted.
- `git revert` restores the environment.

Many things can break those claims: a chart upgrade, a moved upstream
manifest, a malformed digest (one already did), a wrong threshold (one
already did), a change to a script. The check has to run the real thing,
from nothing, often enough that a regression points at one change.

## Decision

`.github/workflows/e2e-gitops.yml` runs on every push to main, on demand,
and as a reusable workflow that `ci.yml` calls for every pull request (the
version under review is tested before it merges). It:

1. creates a kind cluster (Kubernetes 1.35, node image pinned by digest) on
   the runner;
2. runs `make bootstrap-gitops`, the same entry point a person uses. Argo
   CD then installs everything else from the app of apps, and every
   Application must be Synced and Healthy;
3. smoke-tests both environments against the digest their overlay pins;
4. runs the drift check on prod;
5. runs `scripts/demo-canary.sh` on prod. The good release must reach 100%,
   the bad one must be aborted with the stable revision untouched, and the
   revert must end Healthy. Any other outcome fails the run;
6. records the run with VHS.

It uploads the GIF, the transcript, the Rollout, every AnalysisRun with its
measurements, the events, and the state of every Application.

**Releases go through git, on a throwaway branch.** The demonstration must
change the desired state the way a release does, with a commit that Argo CD
picks up. The cluster tracks a branch created at the commit under test
(`e2e/<run>` or `ci/canary-<run>`). The script commits to it in a temporary
worktree and pushes, and the branch is deleted when the run ends. Pushes
made with the workflow token trigger no workflows, so nothing loops.

**The same scripts in every context.** The bootstrap, the smoke test, the
drift check and the demonstration are the scripts a person runs. The
delivery jobs in `release.yml` run them as well. The workflow only
sequences them, so the end-to-end test does not drift from what it tests.

## Alternatives considered

- **A git server inside the cluster (Gitea, soft-serve) or a `file://`
  mirror mounted into Argo CD's repo server.** Either would avoid pushing
  branches to GitHub. But either means Argo CD reads from a different
  source than in every other context: another component to configure and
  pin, plus a repo-server patch only the test uses. A throwaway branch
  exercises the real path: GitHub, polling and a refresh.
- **Moving the Application's `targetRevision` between commits instead of
  committing.** This needs no push rights. But it is a change to the
  cluster, not to git, which is exactly what the drift check shows Argo CD
  undoing.
- **A long-lived test cluster.** It would avoid rebuilding the platform on
  each run (about 15 of the 35 minutes). But it needs an account and
  money, and it accumulates state that a from-scratch run never sees: old
  CRD versions, leftover objects. Bootstrapping from nothing each time also
  tests the bootstrap.
- **k3d or minikube on the runner.** These are equivalent for this
  purpose. kind has the fastest cold start on GitHub runners, its node
  image can be pinned by digest, and its node is a container `docker exec`
  can reach, which is how the model is loaded.
- **Running only on main, or only nightly.** This is cheaper, but a
  regression would then be found after it merged, and possibly after
  several other merges. Running on pull requests puts the 35 minutes
  before the merge, where the cause is one diff.

## Consequences

- Every pull request that is not from Dependabot costs about 35 minutes of
  a runner. That is free on a public repository, and runs in parallel with
  the faster checks.
- Dependabot pull requests skip the job, because they get no repository
  secrets and therefore no sealing key. Their changes are tested by the
  run on main after merging.
- The repository sees short-lived `e2e/*` and `ci/canary-*` branches. A
  cancelled run can leave one behind; it is safe to delete.
- The end-to-end runs on main are the evidence behind the README badge.
