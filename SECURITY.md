# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue: on GitHub,
open the repository's **Security** tab and choose **Report a vulnerability**.
Include what you found, how to reproduce it, and the version (`ascii-magic --version`).

## Supported versions

Fixes go into the latest release. Python 3.11 and newer are supported;
3.10 reached end of life in October 2026.

## What the web GUI is (and isn't) built for

`ascii-magic web` has no login. It is meant for your own machine
(it listens on `127.0.0.1` by default) or a network you trust. Within that,
it treats every request and upload as untrusted: size and work budgets,
cross-site request blocking, sandboxed previews, and a restricted ffmpeg for
uploaded video. See "Server limits" in the README for the settings.
