"""
The version number is obtained from git tags, branch and commit identifier.
It has been designed for the following workflow:

- git init
- create setup.py commit
- more commit
- set version 0.1 in setup.py -> 0.1.dev03
- modify, commit              -> 0.1.dev04
- git checkout -b v0.1        -> 0.1.dev04
- modify, commit              -> 0.1.pre01
- modify, commit              -> 0.1.pre02
- git tag 0.1                 -> 0.1
- modify... and commit        -> 0.1.post01
- modify... and commit        -> 0.1.post02
- git tag 0.1.1               -> 0.1.1
- modify... and commit        -> 0.1.1.post01
- git checkout master         -> 0.1.dev04
- set version=0.2 in setup.py -> 0.2.dev01
- modify, commit              -> 0.2.dev02
- git tag 0.2                 -> 0.2
- set version=0.3 in setup.py -> 0.3.dev01


When working on the master branch, the dev number is the number of commits
since the last release branch (by default of name "v[0-9.]+", but it is
configurable) or the last tag.

"""

# These variables can be changed by the hooks importer
ABBREV = 5
F2PY_TABLE = {'integer': {'int8': 'char',
                          'int16': 'short',
                          'int32': 'int',
                          'int64': 'long_long'},
              'real': {'real32': 'float',
                       'real64': 'double'},
              'complex': {'real32': 'complex_float',
                          'real64': 'complex_double'}}
REGEX_RELEASE = '^v(?P<name>[0-9.]+)$'
try:
    import os
    from Cython.Build import cythonize
    USE_CYTHON = bool(int(os.getenv('SETUPHOOKS_USE_CYTHON', '1') or '0'))
except ImportError:
    USE_CYTHON = False

import numpy
import re
import shutil
import sys
from distutils.command.clean import clean
from numpy.distutils.command.build import build
from numpy.distutils.command.build_src import build_src
from numpy.distutils.command.sdist import sdist
from numpy.distutils.core import Command
from numpy.distutils.misc_util import has_f_sources
from subprocess import call, Popen, PIPE
from warnings import filterwarnings

try:
    root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    root = os.path.dirname(os.path.abspath(sys.argv[0]))

# monkey patch to allow pure and elemental routines in preprocessed
# Fortran libraries
numpy.distutils.from_template.routine_start_re = re.compile(
    r'(\n|\A)((     (\$|\*))|)\s*((im)?pure\s+|elemental\s+)*(subroutine|funct'
    r'ion)\b', re.I)
numpy.distutils.from_template.function_start_re = re.compile(
    r'\n     (\$|\*)\s*((im)?pure\s+|elemental\s+)*function\b', re.I)


def get_cmdclass():

    class BuildCommand(build):
        def run(self):
            _write_version(self.distribution.get_name(),
                           self.distribution.get_version())
            build.run(self)

    class BuildSrcCommand(build_src):
        def initialize_options(self):
            build_src.initialize_options(self)
            self.f2py_opts = '--quiet'

        def run(self):
            has_fortran = False
            has_cython = False
            for ext in self.extensions:
                has_fortran = has_fortran or has_f_sources(ext.sources)
                for isource, source in enumerate(ext.sources):
                    if source.endswith('.pyx'):
                        if not USE_CYTHON:
                            ext.sources[isource] = source[:-3] + 'c'
                        else:
                            has_cython = True
            if has_fortran:
                with open(os.path.join(root, '.f2py_f2cmap'), 'w') as f:
                    f.write(repr(F2PY_TABLE))
            if has_cython:
                build_dir = None if self.inplace else self.build_src
                new_extensions = cythonize(self.extensions, force=True,
                                           build_dir=build_dir)
                for i in range(len(self.extensions)):
                    self.extensions[i] = new_extensions[i]
            build_src.run(self)

    class SDistCommand(sdist):
        def make_release_tree(self, base_dir, files):
            _write_version(self.distribution.get_name(),
                           self.distribution.get_version())
            initfile = os.path.join(self.distribution.get_name(),
                                    '__init__.py')
            new_files = []
            for f in files:
                if f.endswith('.pyx'):
                    new_files.append(f[:-3] + 'c')
            if initfile not in files:
                new_files.append(initfile)
            files.extend(new_files)
            sdist.make_release_tree(self, base_dir, files)

    class CleanCommand(clean):
        def run(self):
            clean.run(self)
            try:
                print(run_git('clean -fdX' + ('n' if self.dry_run else '')))
                return
            except RuntimeError:
                pass

            extensions = '.o', '.pyc', 'pyd', 'pyo', '.so'
            for root_, dirs, files in os.walk(root):
                for f in files:
                    if os.path.splitext(f)[-1] in extensions:
                        self.__delete(os.path.join(root_, f))
                for d in dirs:
                    if d in ('build', '__pycache__'):
                        self.__delete(os.path.join(root_, d), dir=True)
            files = (
                'MANIFEST',
                os.path.join(self.distribution.get_name(), '__init__.py'))
            for f in files:
                if os.path.exists(f):
                    self.__delete(f)

        def __delete(self, file_, dir=False):
            msg = 'would remove' if self.dry_run else 'removing'
            try:
                if not self.dry_run:
                    if dir:
                        shutil.rmtree(file_)
                    else:
                        os.unlink(file_)
            except OSError:
                msg = 'problem removing'
            print(msg + ' {!r}'.format(file_))

    class CoverageCommand(Command):
        description = "run the package coverage"
        user_options = [('file=', 'f', 'restrict coverage to a specific file'),
                        ('erase', None,
                         'erase previously collected coverage before run'),
                        ('html-dir=', None,
                         'Produce HTML coverage information in dir')]

        def run(self):
            cmd = [sys.executable, '-mnose', '--with-coverage', '--cover-html',
                   '--cover-package=' + self.distribution.get_name(),
                   '--cover-html-dir=' + self.html_dir]
            if self.erase:
                cmd.append('--cover-erase')
            call(cmd + [self.file])

        def initialize_options(self):
            self.file = 'test'
            self.erase = 0
            self.html_dir = 'htmlcov'

        def finalize_options(self):
            pass

    class TestCommand(Command):
        description = "run the test suite"
        user_options = [('file=', 'f', 'restrict test to a specific file')]

        def run(self):
            call([sys.executable, '-mnose', self.file])

        def initialize_options(self):
            self.file = 'test'

        def finalize_options(self):
            pass

    return {'build': BuildCommand,
            'build_src': BuildSrcCommand,
            'clean': CleanCommand,
            'coverage': CoverageCommand,
            'sdist': SDistCommand,
            'test': TestCommand}


def get_version(name, default):
    return _get_version_git(default) or _get_version_init_file(name) or default


def run_git(cmd, cwd=root):
    git = 'git'
    if sys.platform == 'win32':
        git = 'git.cmd'
    cmd = git + ' ' + cmd
    process = Popen(cmd.split(), cwd=cwd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        if stderr != '':
            stderr = '\n' + stderr.decode('utf-8')
        raise RuntimeError(
            'Command failed (error {}): {}{}'.format(
                process.returncode, cmd, stderr))
    return stdout.strip().decode('utf-8')


def _get_version_git(default):
    INF = 2147483647

    def get_branches():
        return run_git('for-each-ref --sort=-committerdate --format=%(refname)'
                       ' refs/heads refs/remotes/origin').splitlines()

    def get_branch_name():
        branch = run_git('rev-parse --abbrev-ref HEAD')
        if branch != 'HEAD':
            return branch
        branch = run_git('branch --no-color --contains HEAD').splitlines()
        return branch[min(1, len(branch)-1)].strip()

    def get_description():
        branch = get_branch_name()
        try:
            description = run_git('describe --abbrev={} --tags'.format(ABBREV))
        except RuntimeError:
            description = run_git(
                'describe --abbrev={} --always'.format(ABBREV))
            regex = r"""^
            (?P<commit>.*?)
            (?P<dirty>(-dirty)?)
            $"""
            m = re.match(regex, description, re.VERBOSE)
            commit, dirty = (m.group(_) for _ in 'commit,dirty'.split(','))
            return branch, '', INF, commit, dirty

        regex = r"""^
        (?P<tag>.*?)
        (?:-
            (?P<rev>\d+)-g
            (?P<commit>[0-9a-f]{5,40})
        )?
        (?P<dirty>(-dirty)?)
        $"""
        m = re.match(regex, description, re.VERBOSE)
        tag, rev, commit, dirty = (m.group(_)
                                   for _ in 'tag,rev,commit,dirty'.split(','))
        if rev is None:
            rev = 0
            commit = ''
        else:
            rev = int(rev)
        return branch, tag, rev, commit, dirty

    def get_rev_since_branch(branch):
        try:
            # get best common ancestor
            common = run_git('merge-base HEAD ' + branch)
        except RuntimeError:
            return INF  # no common ancestor, the branch is dangling
        return int(run_git('rev-list --count HEAD ^' + common))

    def get_rev_since_any_branch():
        if REGEX_RELEASE.startswith('^'):
            regex = REGEX_RELEASE[1:]
        else:
            regex = '.*' + REGEX_RELEASE
        regex = '^refs/(heads|remotes/origin)/' + regex

        branches = get_branches()
        for branch in branches:
            # filter branches according to BRANCH_REGEX
            if not re.match(regex, branch):
                continue
            rev = get_rev_since_branch(branch)
            if rev > 0:
                return rev
        # no branch has been created from an ancestor
        return INF

    try:
        run_git('rev-parse --is-inside-work-tree')
    except (OSError, RuntimeError):
        return ''

    branch, tag, rev_tag, commit, dirty = get_description()

    # check if the commit is tagged
    if rev_tag == 0:
        return tag + dirty

    # if the current branch is master, look up the closest tag or the closest
    # release branch rev to get the dev number otherwise, look up the closest
    # tag or the closest master rev.
    suffix = 'dev'
    if branch == 'master':
        rev_branch = get_rev_since_any_branch()
        name = default
        is_branch_release = False
    else:
        rev_branch = get_rev_since_branch('master')
        name = branch
        m = re.match(REGEX_RELEASE, branch)
        is_branch_release = m is not None
        if is_branch_release:
            try:
                name = m.group('name')
            except IndexError:
                pass
        elif rev_tag == rev_branch:
            tag = branch

    if rev_branch == rev_tag == INF:
        # no branch and no tag from ancestors, counting from root
        rev = int(run_git('rev-list --count HEAD'))
        if branch != 'master':
            suffix = 'rev'
    elif rev_tag <= rev_branch:
        rev = rev_tag
        if branch != 'master':
            name = tag
        if is_branch_release:
            suffix = 'post'
    else:
        rev = rev_branch
        if is_branch_release:
            suffix = 'pre'
    if name != '':
        name += '.'
    return '{}{}{:02}{}'.format(name, suffix, rev, dirty)


def _get_version_init_file(name):
    try:
        f = open(os.path.join(name, '__init__.py')).read()
    except IOError:
        return ''
    m = re.search(r"__version__ = '(.*)'", f)
    if m is None:
        return ''
    return m.groups()[0]


def _write_version(name, version):
    try:
        init = open(os.path.join(root, name, '__init__.py.in')).readlines()
    except IOError:
        return
    init += ['\n', '__version__ = ' + repr(version) + '\n']
    open(os.path.join(root, name, '__init__.py'), 'w').writelines(init)


filterwarnings('ignore', "Unknown distribution option: 'install_requires'")
