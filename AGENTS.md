# Project-wide working rules

## Shut down servers when finished

At the end of every task, stop all development, preview, dashboard, and test servers started for this project, including background processes. Do not leave a server running after the final response unless the user explicitly requests it for that task.

Before finishing, check for remaining project server processes and listening ports. Stop any confirmed project servers still running and verify their shutdown. Do not stop unrelated applications or servers. If permissions prevent verification or shutdown, clearly report that limitation.

Prefer serving previews in a tracked foreground session so they can be stopped reliably. A browser tab remaining open does not mean its server should remain running.
