configuring the sphinx is not terrible if you choose the right way.

installation
-------------------
pip install Sphinx
or something similar

main fail points
--------------------
1. it's better to use sphinx-apidoc because it automatically creates the documentation structure.
   DO NOT USE sphinx-quickstart
   
2. sphinx have some extensions. the main ones we need are numpydoc, and autodoc

3. after we set everything up, the directory in which we create the html documentation should not be inculded in the git, or other sub-version control.

sphinx syntax with numpydoc
------------------------------
https://pythonhosted.org/an_example_pypi_project/sphinx.html#python-cross-referencing-syntax
http://bwanamarko.alwaysdata.net/napoleon/format_exception.html#numpydoc
https://github.com/numpy/numpy/blob/master/doc/example.py
https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#a-guide-to-numpyscipy-documentation
http://codeandchaos.wordpress.com/2012/08/09/sphinx-and-numpydoc/
http://codeandchaos.wordpress.com/2012/07/30/sphinx-autodoc-tutorial-for-dummies/

efective sources
---------------------
http://raxcloud.blogspot.co.il/2013/02/documenting-python-code-using-sphinx.html
http://sphinx-doc.org/man/sphinx-apidoc.html

extensions
---------------
http://sphinx-doc.org/ext/todo.html

some additional sources, but you will not really need to get into this
-------------------------------------------------------------------------
http://scriptsonscripts.blogspot.co.il/2012/09/quick-sphinx-documentation-for-python.html
http://sphinx-doc.org/tutorial.html
http://sphinx-doc.org/ext/autodoc.html#module-sphinx.ext.autodoc
https://pythonhosted.org/an_example_pypi_project/sphinx.html
