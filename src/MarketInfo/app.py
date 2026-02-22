"""
Main Application Shell
Handles nav + page routing only
"""

from dash import Dash, html, dcc, page_container, page_registry, clientside_callback, Input, Output
from config import Colors, Typography, Effects, APP_HOST, APP_PORT, DEBUG_MODE

app = Dash(__name__, use_pages=True)
app.title = "Stock Dashboard"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes slideInRight {
                from { opacity: 0; transform: translateX(40px); }
                to   { opacity: 1; transform: translateX(0); }
            }
            @keyframes slideInUp {
                from { opacity: 0; transform: translateY(16px); }
                to   { opacity: 1; transform: translateY(0); }
            }
            * {
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            body {
                margin: 0;
                overflow-x: hidden;
                background: #0a0a0f;
            }
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #13131a; }
            ::-webkit-scrollbar-thumb { background: #3a3a48; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #00f5ff; }
            .btn-hidden { display: none !important; }
            .btn-visible { display: flex !important; }
            .nav-link {
                color: #606078;
                text-decoration: none;
                padding: 6px 18px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                transition: all 0.2s ease;
                border: 1px solid transparent;
                font-family: 'Outfit', sans-serif;
            }
            .nav-link:hover {
                color: #ffffff;
                background: rgba(0, 245, 255, 0.08);
                border-color: rgba(0, 245, 255, 0.25);
            }
            .nav-link-active {
                color: #00f5ff !important;
                background: rgba(0, 245, 255, 0.1) !important;
                border-color: rgba(0, 245, 255, 0.4) !important;
            }
            .page-content {
                animation: slideInRight 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94) both;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Nav bar
nav = html.Div([
    html.Link(
        href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap",
        rel="stylesheet"
    ),
    html.Div([
        # Logo
        html.Div([
            html.Span("◈", style={"color": "#00f5ff", "fontSize": "18px"}),
            html.Span("STOCKDASH", style={
                "color": "#ffffff",
                "fontWeight": "800",
                "fontSize": "13px",
                "letterSpacing": "0.15em",
                "fontFamily": "'Outfit', sans-serif"
            })
        ], style={"display": "flex", "alignItems": "center", "gap": "10px", "marginRight": "32px"}),

        # Nav links (auto-generated from page registry)
        html.Div([
            dcc.Link(
                page["name"],
                href=page["path"],
                className="nav-link",
            )
            for page in page_registry.values()
        ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
    ], style={
        "maxWidth": "1800px",
        "margin": "0 auto",
        "display": "flex",
        "alignItems": "center",
        "padding": "0 32px",
        "height": "100%",
    })
], style={
    "background": "rgba(10, 10, 15, 0.95)",
    "backdropFilter": "blur(16px)",
    "WebkitBackdropFilter": "blur(16px)",
    "borderBottom": "1px solid #2a2a38",
    "height": "52px",
    "position": "sticky",
    "top": "0",
    "zIndex": "1000",
    "display": "flex",
    "alignItems": "center",
})

app.layout = html.Div([
    nav,
    html.Div(page_container, id="page-content", className="page-content")
], style={"background": "#0a0a0f", "minHeight": "100vh"})

# Re-trigger page transition animation on URL change
clientside_callback(
    """
    function(pathname) {
        const el = document.getElementById('page-content');
        if (el) {
            el.style.animation = 'none';
            el.offsetHeight;
            el.style.animation = 'slideInRight 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94) both';
        }
        return pathname;
    }
    """,
    Output("page-content", "data-pathname"),
    Input("_pages_location", "pathname"),
)

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=DEBUG_MODE)