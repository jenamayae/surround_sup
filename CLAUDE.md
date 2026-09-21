# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Standing Rules

- Only edit the files named in the task. Don't create new files unless the task says to.
- Do not rename, reformat, reorganize, or add abstractions, error handling,
  tests, scripts, CLI options, or features that weren't asked for.
- Do not delete code unless the task says to.
- If something outside the task seems to need changing, stop and ask.
- Before editing, list the functions you will change and wait for approval.
- Timing is defined in frames at refresh_hz = 120. center_flicker_hz and
  surround_flicker_hz are reversal rates (reversals per second), not full
  cycles. Don't change that convention.