# How to contribute

I'm really glad you're reading this. I assume you are here because you want to contribute to this code. Any help and suggestions are welcome, and credit will always be given. 😊


## Do you have a question?

Please use the [discussions section](https://github.com/regeirk/pycwt/discussions) to ask questions, share ideas regarding this library or wavelet analysis in general. In the vast majority of cases discussions are better than issues -- you should only open issues if you are sure you found a bug that is reproducible.


## Did you find a bug?

Please report a bug by creating a [new issue](https://github.com/regeirk/pycwt/issues). Make sure that the same bug has not been reporterd before by doing a quick search in the open issues.

Please DO NOT create a new issue for general questions or comments related to wavelet analysis like:

- Reconstructed waveform from `icwt` is still different from the original timeseries.
- The DOF for the global wavelet spectrum will change if scales are in days instead of years?


## Did you write a patch that fixes a bug?

You are welcome and free to contribute to the code and make pull requests to the development branch to fix any issue or enhance the code. You don't need to open an issue first, but if your patch fixes a listed issue, please mention the issue number in the commit message and eventually close it.


## Submitting changes

Please add any changes to the `development` branch with a clear and succint description of what you've done.

Always write a clear log message for your commits and please follow the [Conventional Commits guidelines](https://www.conventionalcommits.org/).


Coding conventions
------------------

Of course you follow coding conventions best practices such as PEP 8, for example. [This here](https://numpydoc.readthedocs.io/en/latest/format.html) is a nice style guide to follow. Keep in mind that you can always use tools like [`black`](https://black.readthedocs.io) and [`isort`](https://isort.readthedocs.io) to ensure the code looks good.
