#!/usr/bin/env python3
"""
Example utility script for writing-requirements.

Replace with actual implementation or delete if not needed.
Scripts are preferred over asking Claude to generate code because they are
more reliable, save tokens, and ensure consistency.
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: writing-requirements_helper.py <input>")
        sys.exit(1)

    input_path = sys.argv[1]
    print(f"Processing: {input_path}")
    # TODO: Add actual logic here


if __name__ == "__main__":
    main()
