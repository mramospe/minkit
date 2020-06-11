# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS       =
SPHINXBUILD      = python -msphinx
SPHINXPROJ       = minkit
SOURCEDIR        = source
BUILDDIR        := $(if $(BUILDDIR),$(BUILDDIR),../../minkit-docs)
NOTEBOOKS_DIR    = source/notebooks
GITCOMMANDS      = cd $(BUILDDIR)/html; git add . ; git commit -m 'Rebuilt docs' ; git push origin gh-pages
PROJECT_SOURCES  = $(shell find ../minkit -type f)

ALLNOTEBOOKS = $(wildcard $(NOTEBOOKS_DIR)/*.ipynb)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo 'If the target is "gh-pages", "html" will be built, and the '\
	'result will be automatically published through the command'
	@echo "$(GITCOMMANDS)"

.PHONY: help Makefile

# Target to build the GitHub Pages
gh-pages: html
	$(GITCOMMANDS)

# Target to update the notebooks. Notebooks are rebuilt in-place, so to avoid
# circular dependencies the target runs every time we call it.
%.ipynb: FORCE
	jupyter nbconvert --execute --to notebook --output $(subst $(NOTEBOOKS_DIR)/,,$@) --output-dir $(NOTEBOOKS_DIR) $@

# Target for Jupyter notebooks
notebooks: $(ALLNOTEBOOKS);

# Forces running certain rules
FORCE: ;

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html: Makefile $(PROJECT_SOURCES) notebooks
	@if [ -d $(BUILDDIR)/doctrees ]; then rm -r $(BUILDDIR)/doctrees; fi
	@if [ -d source/api ]; then rm -r source/api; fi
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Only to be used when the notebooks have already been created and only
# parts of the code not related to the API are modified.
html-no-notebooks: Makefile $(PROJECT_SOURCES)
	@if [ -d $(BUILDDIR)/doctrees ]; then rm -r $(BUILDDIR)/doctrees; fi
	@if [ -d source/api ]; then rm -r source/api; fi
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
