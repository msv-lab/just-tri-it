# README

## 1. Install elan(Version Manager)

elan is a version manager for Lean.

**Installation command**

```
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash
```

Choose the default installation path (default: ~/.elan).

**Check Installation:**

```
elan --version
```

Ensure that the version is ≥ 2.0.0.

## 2.Install Lean Toolchain

Navigate to the project directory (e.g., proofs):

```
cd proofs
```

Since the lean-toolchain file already exists, running the following command will automatically install the Lean version corresponding to the toolchain:

```
lean --version
```

Check whether lake version matches:

```
lake --version
```

## 3. Build project

```
lake update
lake build
```

if it shows：

```
Build completed successfully
```

Indicates that the build was successful!

