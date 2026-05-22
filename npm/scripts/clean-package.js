#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

const packageRoot = path.resolve(__dirname, "..");

const entries = ["README.md", "LICENSE.txt", "docs"];

for (const entry of entries) {
  fs.rmSync(path.join(packageRoot, entry), { recursive: true, force: true });
}
