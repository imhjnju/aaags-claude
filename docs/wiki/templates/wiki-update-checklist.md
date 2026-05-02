# Wiki Update Checklist

Use this whenever editing `docs/wiki/`.

## Scope

- [ ] What changed?
- [ ] Which readers are affected?
- [ ] Is wiki the right layer, or should this be spec/memory/dev_notes instead?

## Authority

- [ ] Does the page identify the authoritative source?
- [ ] Does it avoid duplicating volatile state?
- [ ] If it includes metrics/counts, are they dated or linked to living status?

## Links

- [ ] Relative links resolve inside `docs/wiki/`.
- [ ] Links to repo files point to stable source-of-truth files.
- [ ] Historical docs are labeled or linked with caution.

## Content Quality

- [ ] The page answers a real reader question.
- [ ] The page helps both humans and agents.
- [ ] It contains no temporary task chatter.
- [ ] It avoids repeating large blocks from existing docs.
- [ ] It has actionable checklists where useful.

## Reader Test

Ask a fresh reader:

- [ ] What is this page for?
- [ ] What should I do next after reading it?
- [ ] Which source is authoritative?
- [ ] What could be ambiguous?

## Final

- [ ] Update `README.md` index if a page was added/removed.
- [ ] Update `dev_notes/session_state.md` if wiki change is milestone-relevant.
- [ ] If the user explicitly requested a commit: check status/diff, exclude artifacts, complete required review, then commit with a message explaining why the wiki changed.
