# Before reporting or releasing results

- [ ] Freeze protocol, code commit, data checksum, model revisions, and environment.
- [ ] Execute a clean smoke run, then the full paper profile on documented hardware.
- [ ] Retain run manifests, failures/skips, per-query measurements, and artifact hashes.
- [ ] Explain skipped IVFPQ experiments and actual unique corpus sizes.
- [ ] Use independent human judgments; report pool design, count, agreement, and adjudication.
- [ ] Report paired query intervals separately per seed; distinguish exploratory analyses.
- [ ] Keep proxy, ANN recall, classifier accuracy, and human relevance metrics distinct.
- [ ] Report inference stages and latency distributions; avoid unsupported real-time claims.
- [ ] Regenerate every manuscript table/figure from one identified run collection.
- [ ] Reconcile all percentages, MRR directions, dataset counts, and method descriptions.
- [ ] Remove unimplemented claims, including cross-validation or sentiment-concatenated ablations.
- [ ] Explain target-domain validation, language/model confounds, binary-label exclusions,
      review-as-query limits, corpus metadata errors, and explicit sentiment-intent limits.
- [ ] Add verified related work and a comparison table with compatible evaluation scopes.
- [ ] Typeset compact tables in the venue template and rewrite conclusions from measured evidence.
- [ ] Review dataset/model redistribution terms; release derived artifacts only when allowed.
- [ ] Add actual author names, repository URL, release version, and optional archived DOI.
- [ ] Clear notebook outputs and review staged files before making the GitHub repository public.
