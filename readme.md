# Thermal-Aware On-Device Training

## Setup

``` shell
# clone the latest MNN (of our fork) into the `main` folder
$ mkdir mnn
$ cd mnn
mnn$ git clone https://github.com/acies-os/MNN.git main
mnn$ cd main

# add Melon's code as a subtree
mnn/main$ git remote add melon https://github.com/acies-os/Melon.git
mnn/main$ git fetch melon
mnn/main$ git worktree add ../melon-main melon/main

# add the version of MNN that Melon is based on as a subtree
mnn/main$ git worktree add -b acies/melon-base ../melon-base 6a6d1a095d0cd573bb67ba9608e6c686ca11c834

# Now our folder should look like this:
#
# mnn/
# ├── main/
# ├── melon-base/
# └── melon-main/
```

To view the diff between `melon-base` and `melon-main`:

``` shell
mnn/melon-base$ git cherry-pick melon/main
# then you can view the changes in your editor, e.g.:
mnn/melon-base$ code .
# changed files without conflicts will be staged
# changes with conflicts will be in the git merge conflict format
```

Conventions:

- Prefix branch names with `acies/`. For example, while working on this readme, I create a branch `acies/readme`.

If you need to create another worktree:

``` shell
# Assume you're in `main` folder and need to work on another feature temporarily
mnn/main$ git worktree add -b acies/<name> ../feat-<name> <hash>
# this should result in:
#
# mnn/
# ├── main/
# ├── feat-<name>/
# ├── melon-base/
# └── melon-main/
mnn/main$ cd ..
mnn$ code feat-<name>
# go work on `feat-<name>`

# once you're done (feature pushed and merged), you can delete the worktree
mnn/feat-<name>$ cd ../main # go back to the main worktree
mnn/main$ git worktree remove ../feat-<name>
```
