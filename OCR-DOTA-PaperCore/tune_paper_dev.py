import sys
from paper_campaign import main
if __name__ == '__main__':
    sys.argv.extend(['--phase','development'])
    raise SystemExit(main())
