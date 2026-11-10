# VHS tape for the canary GIF at the top of the README.
#
# CI renders it after scripts/demo-canary.sh has run against a real cluster
# (the GitOps job in .github/workflows/ci.yml), by replaying that run's
# transcript as a time-lapse with scripts/replay-log.sh. The first line of
# the recording states the speed-up.
#
#   docker run --rm -v "$PWD:/vhs" ghcr.io/charmbracelet/vhs:v0.12.1 docs/demo/canary.tape
#
# It expects the transcript at demo-canary.log in the working directory.

Output docs/demo/canary.gif

Require bash

Set Shell "bash"
Set FontSize 15
Set Width 1500
Set Height 860
Set Padding 20
Set Framerate 12
Set Theme "Catppuccin Mocha"

Hide
Type "clear && bash scripts/replay-log.sh demo-canary.log 50 'recorded in CI'"
Enter
Sleep 500ms
Show
Sleep 58s
