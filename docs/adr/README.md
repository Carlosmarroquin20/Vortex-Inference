# Architecture decision records

One record per decision that shapes how the service is built, shipped or
operated. Each record states the context, the decision, the alternatives that
were rejected and why, and the consequences. Records are not edited after
acceptance; a later decision supersedes an earlier one in a new record.

| ADR | Decision |
|-----|----------|
| [0001](0001-publish-signed-multi-arch-images.md) | Publish signed multi-architecture images to GHCR |
| [0002](0002-gitops-with-argo-cd.md) | GitOps with Argo CD: staging and prod promoted by commit |
| [0003](0003-canary-releases-with-argo-rollouts.md) | Canary releases with Argo Rollouts and Prometheus analysis |
| [0004](0004-end-to-end-test-on-ephemeral-clusters.md) | End-to-end test on an ephemeral kind cluster, through git |
| [0005](0005-rollback-by-git-revert.md) | Roll back by reverting the release commit |
| [0006](0006-catalog-enrichment-job.md) | Catalog enrichment as a queued job, schema-validated and grammar-constrained |
