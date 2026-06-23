---
layout: post
title: Keep record of archived directory trees with `ncdu` and `zstd`
date: 2026-05-22 10:50:00
description: this is what included plotly.js code could look like
tags: Unix bash archive
categories: how-to
---

# Keeping a record of archived directory trees with `ncdu` + `zstd`
Every now and then IT archives a chunk of my data to a safer drive and tells me I can delete my local copy to free up space. That's fine in principle — but the moment the directory is gone, I lose any sense of *what* was in it and *how* it was organised. So before deleting anything I want a small, browsable record of the tree that I can keep around indefinitely and later use to tell IT "please reinstate `X/Y/Z` from the backup." What makes this approach so pleasant is that the record isn't just a flat text listing: it lets me re-open the directory exactly as it stood at the moment of backup and wander through it interactively — folder by folder, sizes and all — long after the data themselves are gone.

Two tools do this nicely:
- **`ncdu`** ("NCurses Disk Usage") scans a directory and can export the whole tree — every file and folder, with sizes — as a compact JSON dump. Crucially, it can later re-open that dump in its interactive browser *without the original data being present*.
- **`zstd`** (Zstandard) compresses the dump. Directory listings share enormous common prefixes, so they shrink dramatically — a tree of millions of files becomes a file of a few tens of MB at most.

Together they turn an entire directory tree into a small file I can browse, grep, and archive next to the data it describes.
## Setup
I keep `ncdu` in its own conda environment so it doesn't clutter anything else. A single command both creates the environment and installs `ncdu` into it:
```bash
conda create -n ncdu-archive-viewer -c conda-forge ncdu
```

## Capture the tree — *before* IT deletes anything
```bash
ncdu -x -o - /path/to/dir | zstd -19 -f -o backup-index.ncdu.zst
```
What the flags do:
- `-x` keeps the scan on a single filesystem.
- `-o -` writes the export to stdout, which I pipe straight into `zstd`.
- `-19 -f` compresses hard and overwrites any stale file of the same name.

## Convince myself it's all there
```bash
zstdcat backup-index.ncdu.zst | ncdu -f-
```
The command first uses `zstdcat` to decompress the backup file and pipes it to `ncdu` for reading. (The trailing dash in `-f-` means that `ncdu` is reading the file from stdin.) This drops me into `ncdu`'s normal arrow-key browser, except it's a frozen snapshot of a directory that may no longer exist. I poke around, confirm the folders I expect are present with sensible sizes, and only *then* am I happy to delete the original.

## Notes to self (learned the hard way)
- **A scan that "finishes" suspiciously early is a red flag, not a relief.** The danger is a silently truncated index that still looks complete when browsed — I can't see what isn't there.
- **Sanity-check completeness by counting entries.** The number of items in the index, `zstdcat backup-index.ncdu.zst | grep -o '"name":' | wc -l`, should match `find /path/to/dir -xdev | wc -l` (both include the root directory; ncdu's on-screen `Items:` count is exactly one less, which is fine).

That's the whole procedure: one environment, one command to capture, one command to browse. Small file, big peace of mind.

## Appendix: searching the index by regex
To pick a particular file or folder out of the snapshot without clicking through the browser, I use the small Python script at the very end of this file. It walks the export, rebuilds the full path of every entry, and prints the ones whose path matches a regular expression — with sizes, so I can immediately judge what's worth restoring. I save it as `ncdu_grep.py` (see bottom of page) and feed it the decompressed index on stdin.

For example, to hunt down a MATLAB results file — one whose name starts with `result` and ends in `.mat`:
```bash
zstdcat backup-index.ncdu.zst | python3 ncdu_grep.py 'result.*\.mat$'
```
The pattern `result.*\.mat$` is a *regular expression* (not a shell glob), so each symbol carries a specific meaning:
- `.` matches any single character.
- `*` means "zero or more of the item immediately before it", so the pair `.*` matches any run of characters — here, whatever sits between `result` and the extension.
- `\` escapes the character that follows it: `\.` therefore matches a literal dot, instead of the "any character" that a bare `.` would otherwise mean.
- `$` anchors the match to the *end* of the path, so only entries that genuinely end in `.mat` are returned (not, say, `result_old.mat.bak`).

In short: a literal `result`, then any characters, then a literal `.mat` right at the end. You can find the script below:

```python
#!/usr/bin/env python3
"""Search an ncdu JSON export (read on stdin) by regex over full paths.

Usage:
    zstdcat backup-index.ncdu.zst | ncdu_grep.py 'REGEX'
    zstdcat backup-index.ncdu.zst | ncdu_grep.py -i 'session0[12]'
"""
import sys, re, json, posixpath

# Parse command-line arguments, support optional case-insensitive matching.
args = sys.argv[1:]
flags = 0
if args and args[0] == "-i":            # optional case-insensitive flag
    flags = re.IGNORECASE
    args = args[1:]
pat = re.compile(args[0], flags)

# Load the ncdu JSON export from stdin and grab the directory tree.
# The JSON format used by ncdu stores the directory tree as the 4th element.
tree = json.load(sys.stdin)[3]

def human(n):
    """Format a byte count as a human-readable size string.

    The function scales the value through B, K, M, G, T units, using 1024 as
    the scaling factor. If the value exceeds the T range, it falls back to P.
    """
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:6.1f}{u}"
        n /= 1024
    return f"{n:6.1f}P"


def walk(node, prefix):
    """Recursively walk the ncdu directory tree and print matching paths.

    `node` is a directory node from the ncdu JSON tree. The first element is a
    dict with the directory info, and the remaining entries are either child
    directories (lists) or file objects.
    """
    info = node[0]                      # first element describes the dir itself
    path = info["name"] if not prefix else posixpath.join(prefix, info["name"])

    # If the directory path itself matches the regex, print it with trailing '/'.
    if pat.search(path):
        print(f"{human(info.get('dsize', 0))}  {path}/")

    for child in node[1:]:
        if isinstance(child, list):     # a list => subdirectory: recurse
            walk(child, path)
        else:                           # an object => file
            cpath = posixpath.join(path, child["name"])
            if pat.search(cpath):
                print(f"{human(child.get('dsize', 0))}  {cpath}")

# Start recursive traversal from the root of the tree.
walk(tree, "")
```
