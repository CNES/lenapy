Release procedure
=================

Release for lenapy, from within your fork:

* Submit a PR that updates the release notes in `doc/changelog.rst`.

We submit a PR to inform other developers of the pending release, and possibly
discuss its content.

* Once the PR is merged, checkout the main branch:

.. code-block:: none

    git checkout upstream/main


* Create a tag and push to github:

.. code-block:: none

    git tag -a x.x.x -m 'Version x.x.x'
    git push --tags upstream

* The Conda Forge bots should pick up the change automatically within an hour or two. Then follow the instructions from the automatic e-mail that you will receive from Conda Forge, basically:

  - Check that dependencies have not changed.

  - Merge the PR on conda-forge/lenapy-feedstock once tests have passed.
