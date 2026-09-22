# Instructions for the next agent

Read `handoff.md` for project state and `README.md` for the runnable workflow.

## User's standing instructions

- Do not use subagents or delegate work to other agents.
- Do not overengineer. Prefer a small, runnable Python solution.
- Do not change files when the user is only asking a question.
- When downloading packages, install explicitly pinned versions that are at least two weeks old. Do not download the latest package versions. Verify release dates before changing dependencies.

## Project direction

Keep segmentation, exports, previews and collision queries executable through Python scripts in VS Code. The user abandoned the website prototype. Do not introduce a browser frontend, Node.js service, Unity project or large ML installation unless the user asks for that direction.

The long-term objective is individual building assets, ground/roads, rubble and other meaningful objects, each with distinct collision identity for a later Magic Leap 2/robot workflow. The current implementation is an editable, scan-specific baseline. No next feature has been authorized by the handoff alone.

## Implementation rules

- Preserve original face references, UVs, texture bytes and shared coordinates when splitting geometry. Retain uncertain labels where evidence is insufficient.
- Keep inputs read only and generated files under the configured results directory. Do not commit `.venv`, caches or generated result folders.
- Use `run.py` as the main entry point, `config.json` for paths and `annotations.json` for this scan's masks. Do not hardcode a user's absolute paths into code or silently remove the input-hash guard.
- Distinguish provisional visual labels, collision geometry and actual engine/robot integration. Do not call source units meters, treat ground as safe to traverse, or describe the method as a trained automatic segmenter.
- Run checks appropriate to a code change. The default end-to-end check is `python run.py` followed by `python -m unittest discover -s tests -v`. Tests expect the default masks; investigate changes before updating expected counts.

## Communication

Keep updates and answers concise. Lead with what works or the next concrete action. Explain material limitations plainly. Ask for missing information only when it affects the work, and continue independent work where possible.

Keep `handoff.md` current when the implementation, verification status or agreed direction changes. The supplied asset has separate licensing evidence in `source-license.json`; do not assume the local repository is cleared for public asset redistribution.
