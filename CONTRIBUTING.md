############################
Contributing guidelines
############################

We welcome any kind of contribution to our software, from simple comment or question to a full fledged [pull request](https://help.github.com/articles/about-pull-requests/>`_. Please read and follow our `Code of Conduct <CODE_OF_CONDUCT.rst).

A contribution can be one of the following cases:

#. you have a question;
#. you think you may have found a bug (including unexpected behavior);
#. you want to make some kind of change to the code base (e.g. to fix a bug, to add a new feature, to update documentation);
#. you want to make a new release of the code base.

The sections below outline the steps in each case.

You have a question
*******************

#. use the search functionality [here](https://github.com/matchms/matchms/issues) to see if someone already filed the same issue;
#. if your issue search did not yield any relevant results, make a new issue;
#. apply the "Question" label; apply other labels when relevant.

You think you may have found a bug
**********************************

#. use the search functionality [here](https://github.com/matchms/matchms/issues) to see if someone already filed the same issue;
#. if your issue search did not yield any relevant results, make a new issue, making sure to provide enough information to the rest of the community to understand the cause and context of the problem. Depending on the issue, you may want to include:
    - the [SHA hashcode](https://help.github.com/articles/autolinked-references-and-urls/#commit-shas) of the commit that is causing your problem;
    - some identifying information (name and version number) for dependencies you're using;
    - information about the operating system;
#. apply relevant labels to the newly created issue.

You want to make some kind of change to the code base
*****************************************************

#. (**important**) announce your plan to the rest of the community *before you start working*. This announcement should be in the form of a (new) issue;
#. (**important**) wait until some kind of consensus is reached about your idea being a good idea;
#. if needed, fork the repository to your own Github profile and create your own feature branch off of the latest master commit. While working on your feature branch, make sure to stay up to date with the master branch by pulling in changes, possibly from the 'upstream' repository (follow the instructions `here <https://help.github.com/articles/configuring-a-remote-for-a-fork/>`__ and `here <https://help.github.com/articles/syncing-a-fork/>`__);
#. make sure the existing tests still work by running ``python setup.py test``;
#. add your own tests (if necessary);
#. update or expand the documentation;
#. update the `CHANGELOG.rst` file with change;
#. [push](http://rogerdudler.github.io/git-guide/>) your feature branch to (your fork of) the matchms repository on GitHub;
#. create the pull request, e.g. following the instructions [here](https://help.github.com/articles/creating-a-pull-request/).

In case you feel like you've made a valuable contribution, but you don't know how to write or run tests for it, or how to generate the documentation: don't let this discourage you from making the pull request; we can help you! Just go ahead and submit the pull request, but keep in mind that you might be asked to append additional commits to your pull request.

You want to make a new release of the code base
***********************************************

To create release you need write permission on the repository.

1. Check author list in citation.cff and .zenodo.json
2. Bump the version using `bump2version <major|minor|patch>`
3. Update the `CHANGELOG.rst` to include changes made
4. Goto [GitHub release page](https://github.com/matchms/matchms/releases)
5. Press draft a new release button
6. Fill version, title and description field
7. Press the Publish Release button

A GitHub action will run which will publish the new version to anaconda.
Also a Zenodo entry will be made for the release with it's own DOI.
