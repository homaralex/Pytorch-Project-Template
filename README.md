# PyTorch Project Template 2.0

This is a fork of [this repo](https://github.com/moemen95/Pytorch-Project-Template) - go there for a more detailed
README.

Main changes:
 * [gin](https://github.com/google/gin-config) configs instead of json
 * can be installed as an external library allowing code like `from pytorch_template.main import main; main()`
 * removed all examples except MNIST; removed tutorials
 * refactored some of the existing code
 * some small functionalities added, e.g. saving experiment state (config files, archive of the current repo state) on each run

### License:
This project is licensed under MIT License - see the LICENSE file for details
