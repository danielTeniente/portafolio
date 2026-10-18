"""Serve the dashboard in the foreground; closing the console stops it."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import webbrowser


def main():
    docs = Path(__file__).resolve().parents[1] / 'docs'
    if not (docs / 'index.html').is_file():
        print(f'Dashboard not found: {docs}', file=sys.stderr)
        return 1
    handler = partial(SimpleHTTPRequestHandler, directory=str(docs))
    try:
        server = ThreadingHTTPServer(('127.0.0.1', 8000), handler)
    except OSError as exc:
        print(f'Cannot open local port 8000: {exc}', file=sys.stderr)
        print('Close an existing dashboard window before trying again.', file=sys.stderr)
        return 1
    with server:
        print('\nQuito Solar dashboard: http://127.0.0.1:8000', flush=True)
        print('Close this terminal window or press Ctrl+C to STOP the server.', flush=True)
        print('The browser tab may remain open, but the server will be stopped.\n', flush=True)
        try:
            webbrowser.open('http://127.0.0.1:8000')
        except webbrowser.Error:
            print('Open the address above in your browser.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nDashboard stopped.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
