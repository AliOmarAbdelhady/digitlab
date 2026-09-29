const { getDefaultConfig } = require("expo/metro-config");
const path = require("node:path");

const config = getDefaultConfig(__dirname);

// portable model files are shipped as assets (custom extension so metro treats
// them as binary assets instead of compiling giant JSON into the JS bundle)
config.resolver.assetExts.push("dlmodel");

// the shared inference core lives at <repo>/core — make it visible to metro
config.watchFolders = [path.resolve(__dirname, "..", "core")];

module.exports = config;
