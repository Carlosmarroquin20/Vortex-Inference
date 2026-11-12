# 0005. Roll back by reverting the release commit

- Status: accepted
- Date: 2026-11-12

## Context

A release in staging or prod is a commit that writes an image's
`tag@digest` into the environment's overlay (ADR 0002). Argo CD applies it
with self-heal on, and Argo Rollouts releases it as a canary (ADR 0003). A
canary that fails its analysis is aborted automatically: the stable pods
never stop serving. That leaves two cases that need a person:

- a release that passed its analysis and later turns out to be bad, for
  example a slow leak or a defect the analysis does not measure;
- an aborted release, whose commit is still the desired state in git.
  Argo CD keeps reporting the environment Degraded, and the next promotion
  would start from it.

The rollback procedure has to work at 3 a.m. for someone who did not write
the release. It must not be undone by the platform itself, and it must leave
a record.

## Decision

The rollback is a `git revert` of the release commit
(`deploy(<env>): sha-…`), pushed to main like any change:

```bash
git log --oneline -- k8s/overlays/prod/
git revert <commit> && git push origin main
```

Argo CD applies it within a minute. The previous image is still cached on
the nodes, and its pod template is the Rollout's stable revision, so Argo
Rollouts goes back without canary steps. The runbooks and the README point
here; the canary demonstration exercises it on every end-to-end run.

## Alternatives considered

- **`kubectl argo rollouts undo` or `abort`, or `kubectl rollout undo`.**
  They are fast and familiar, but they change the cluster and not git.
  Argo CD's self-heal puts the release back within seconds; the drift check
  in every pipeline run shows exactly that.
- **`argocd app rollback` (Argo CD's history).** It redeploys an earlier
  synced revision, but it requires disabling automated sync first, and the
  environment then runs something git does not describe. Someone has to
  remember to re-enable sync and reconcile git by hand. It is a valid
  emergency lever, not the procedure.
- **Pinning the previous image with a new forward commit.** This reaches
  the same state as a revert. However, it loses the link to the release it
  undoes, and it invites typing a digest by hand. `git revert` names the
  commit it reverses, and its diff can be reviewed in a few seconds.
- **Disabling self-heal so cluster-side rollbacks stick.** This would make
  every manual change permanent and silent, which is the drift GitOps
  exists to prevent.

## Consequences

- A rollback is reviewed, attributed and recorded in the same history as
  the release. The deployment log (`git log -- k8s/overlays/<env>/`) shows
  both.
- It is as fast as Argo CD's poll (about a minute, or immediately with a
  refresh), plus a few seconds for the stable pods, which are already
  running in a passed release's case or never left in an aborted one's.
- In an emergency before a revert can merge, pausing automated sync on the
  Application is the documented escape hatch. It is visible in Argo CD, and
  re-enabling sync returns the environment to git.
