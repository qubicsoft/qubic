"""
The version number is obtained from git tags, branch and commit identifier.
It has been designed for the following workflow:

- git checkout master
- modify, commit, commit
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
- modify, commit              -> 0.1.dev02

When working on the master branch, the dev number is the number of commits
since the last branch of name "v[0-9.]+"

"""
import os
import re
import sys
from numpy.distutils.command.build import build
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.command.sdist import sdist
from numpy.distutils.core import Command
from subprocess import call, Popen, PIPE
from warnings import filterwarnings

try:
    root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    root = os.path.dirname(os.path.abspath(sys.argv[0]))
ABBREV = 5
BRANCH_REGEX = '^refs/(heads|remotes/origin)/v[0-9.]+$'


def get_version(name, default):
    version = get_version_git(default)
    if version != '':
        return version
    return get_version_init_file(name) or default


def get_version_git(default):
    def run(cmd, cwd=root):
        git = "git"
        if sys.platform == "win32":
            git = "git.cmd"
        process = Popen([git] + cmd, cwd=cwd, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        if stderr != '':
            raise RuntimeError(stderr)
        if process.returncode != 0:
            raise RuntimeError('Error code: {0}.'.format(process.returncode))
        return stdout.strip()

    def get_branches():
        return run(['for-each-ref', '--sort=-committerdate', '--format=%(ref'
                    'name)', 'refs/heads', 'refs/remotes/origin']).split('\n')

    def get_branch_name():
        return run(['rev-parse', '--abbrev-ref', 'HEAD'])

    def get_description():
        try:
            description = run([
                'describe', '--tags', '--abbrev={0}'.format(ABBREV)])
        except RuntimeError:
            description = run([
                'describe', '--tags', '--abbrev={0}'.format(ABBREV),
                '--always']).split('-')
            return '', '', description[0], '-' + description[1]
        regex = r"""^
        (?P<tag>.*?)
        (?:-
            (?P<rev>\d+)-g
            (?P<commit>[0-9a-f]{5,40})
        )?
        $"""
        m = re.match(regex, description, re.VERBOSE)
        tag, rev, commit = (m.group(_) for _ in 'tag,rev,commit'.split(','))
        rev = int(rev)
        return tag, rev, commit

    def get_rev_since_branch(branch):
        common = run(['merge-base', 'HEAD', branch])
        return int(run(['rev-list', '--count', 'HEAD', '^' + common]))

    def get_dirty():
        return '-dirty' if run(['diff-index', 'HEAD']) else ''

    def get_master_rev(default):
        branches = get_branches()
        for branch in branches:
            # filter branches according to BRANCH_REGEX
            if not re.match(BRANCH_REGEX, branch):
                continue
            rev = get_rev_since_branch(branch)
            if rev > 0:
                return rev
        return int(run(['rev-list', '--count', 'HEAD']))

    try:
        run(['rev-parse', '--is-inside-work-tree'])
    except (OSError, RuntimeError):
        return ''

    dirty = get_dirty()

    # check if HEAD is tagged
    try:
        return run(['describe', '--tags', '--candidates=0']) + dirty
    except RuntimeError:
        pass

    # if the current branch is master, look up the last release branch
    # to get the dev number
    branch = get_branch_name()
    if get_branch_name() == 'master':
        rev = get_master_rev(default)
        commit = run(['rev-parse', '--short={}'.format(ABBREV), 'HEAD'])
        if default != '':
            return '{}.dev{:02}-g{}{}'.format(default, rev, commit, dirty)
        return str(rev) + dirty

    isrelease = re.match('^v[0-9.]+$', branch) is not None
    rev_master = get_rev_since_branch('master')
    tag, rev_tag, commit = get_description()
    if isrelease:
        version = tag
    else:
        version = branch
    if rev_tag > 0:
        if rev_master < rev_tag:
            version += '.pre{:02}'.format(rev_master)
        else:
            version += '.post{:02}'.format(rev_tag)
        version += '-g' + commit
    return version + dirty


def get_version_init_file(name):
    try:
        f = open(os.path.join(name, '__init__.py')).read()
    except IOError:
        return ''
    m = re.search(r"__version__ = '(.*)'", f)
    if m is None:
        return ''
    return m.groups()[0]


def write_version(name, version):
    try:
        init = open(os.path.join(root, name, '__init__.py.in')).readlines()
    except IOError:
        return
    init += ['\n', '__version__ = ' + repr(version) + '\n']
    open(os.path.join(root, name, '__init__.py'), 'w').writelines(init)


class BuildCommand(build):
    def run(self):
        write_version(self.distribution.get_name(),
                      self.distribution.get_version())
        build.run(self)


class SDistCommand(sdist):
    def make_release_tree(self, base_dir, files):
        write_version(self.distribution.get_name(),
                      self.distribution.get_version())
        sdist.make_release_tree(self, base_dir, files)


class CoverageCommand(Command):
    description = "run the package coverage"
    user_options = [('file=', 'f', 'restrict coverage to a specific file'),
                    ('erase', None,
                     'erase previously collected coverage before run'),
                    ('html-dir=', None,
                     'Produce HTML coverage information in dir')]

    def run(self):
        cmd = ['nosetests', '--with-coverage', '--cover-html',
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
        call(['nosetests', self.file])

    def initialize_options(self):
        self.file = 'test'

    def finalize_options(self):
        pass


def get_cmdclass():
    return {'build': BuildCommand,
            'build_ext': build_ext,
            'coverage': CoverageCommand,
            'sdist': SDistCommand,
            'test': TestCommand}

filterwarnings('ignore', "Unknown distribution option: 'install_requires'")
