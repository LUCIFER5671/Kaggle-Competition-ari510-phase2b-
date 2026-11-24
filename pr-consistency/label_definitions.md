The PRs are agent-authored pull requests (OpenAI Codex, Devin, GitHub Copilot, Cursor, Claude Code) selected because they
perform Extract/Move refactorings (Extract/Move Method, Extract/Move Class, Move Field).

- Consistent: the title and description accurately describe the code changes.
- Inconsistent: the description misses major changes, claims changes that aren't there, or misdescribes them.
  (For example: claims a simple refactoring while also changing logic, leaves out many of the files it touches,
  or describes work that never shows up in the diff.)
- Out of scope: the PR isn't really an Extract/Move refactoring.

(Source: the competition's Overview page. A fourth label, Unavailable - the PR no longer exists - is handled separately
and is not one of your choices.)
