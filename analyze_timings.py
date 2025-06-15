import re
import sys
import html
import json
from pathlib import Path
import math

def parse_duration(s):
    """Parses a duration string like '1.234ms' or '567.8µs' or '1.2s' into milliseconds."""
    s = s.strip().replace(',', '') # handle thousands separators
    
    # Use regex to find the numeric part and the unit
    match = re.match(r'([\d.]+)\s*([a-zA-Zµ]+)', s)
    if not match:
        return 0.0

    val_str, unit = match.groups()
    val = float(val_str)
    
    if unit == 'ns':
        return val / 1_000_000.0
    if unit == 'µs' or unit == 'us':
        return val / 1_000.0
    if unit == 'ms':
        return val
    if unit == 's':
        return val * 1000.0
    return 0.0

def generate_color(time_ms):
    """Generates a color from green to red based on time in ms."""
    if time_ms is None:
        return 'rgba(128, 128, 128, 0.2)' # Grey for no data
    
    # Use a log scale for better color distribution.
    # Let's say 1ms is green, 100ms is red.
    log_min = 0 # log10(1)
    log_max = 2 # log10(100)
    
    if time_ms <= 1:
        log_time = log_min
    else:
        log_time = math.log10(time_ms)
    
    # Normalize to 0-1 range
    ratio = max(0, min(1, (log_time - log_min) / (log_max - log_min)))
    
    # HSL color: Hue goes from 120 (green) down to 0 (red)
    h = 120 * (1 - ratio)
    s = 80
    l = 60
    return f'hsla({h}, {s}%, {l}%, 0.6)'

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <path_to_source_code> <path_to_log_file>")
        sys.exit(1)

    source_path = Path(sys.argv[1])
    log_path = Path(sys.argv[2])
    output_path = Path("timings_visualization.html")

    full_text = source_path.read_text(encoding='utf-8')
    log_content = log_path.read_text(encoding='utf-8')

    # Find all "Processing token..." markers to split the log
    processing_markers = list(re.finditer(r'Processing token \d+/\d+', log_content))
    
    initial_log_block = ""
    if processing_markers:
        initial_log_block = log_content[:processing_markers[0].start()]

    tokens_data = []
    for i, marker in enumerate(processing_markers):
        start_pos = marker.start()
        end_pos = processing_markers[i+1].start() if i + 1 < len(processing_markers) else len(log_content)
        
        token_log = log_content[start_pos:end_pos]
        
        # The get_mask for the first token is in the initial block before the loop
        log_for_get_mask = initial_log_block if i == 0 else token_log

        token_info = {}
        
        m = re.search(r"Processing token \d+/\d+: (.*?)\(LLMTokenID\((\d+)\)\)", token_log)
        if m:
            token_str_raw = m.group(1).strip()
            if token_str_raw.startswith("'") and token_str_raw.endswith("'"):
                token_str_raw = token_str_raw[1:-1]
            elif token_str_raw.startswith('"') and token_str_raw.endswith('"'):
                 token_str_raw = token_str_raw[1:-1]
            token_info['str'] = bytes(token_str_raw, "utf-8").decode("unicode_escape", "replace")
            token_info['id'] = int(m.group(2))

        m = re.search(r"Context Highlight \(Token bytes \[(\d+), (\d+)\)\):", token_log)
        if m:
            token_info['start_byte'] = int(m.group(1))
            token_info['end_byte'] = int(m.group(2))

        get_mask_time = None
        m_get_mask = re.search(r"get_mask took:\s*(.*?)\n", log_for_get_mask)
        if m_get_mask:
            get_mask_time = parse_duration(m_get_mask.group(1))

        commit_time = None
        m_commit = re.search(r"commit LLMTokenID\(\d+\) took:\s*(.*?)\n", token_log)
        if m_commit:
            commit_time = parse_duration(m_commit.group(1))
            
        special_map_time = None
        m_special_map = re.search(r"after special_map:\s*(.*?)\n", log_for_get_mask)
        if m_special_map:
            special_map_time = parse_duration(m_special_map.group(1))

        token_info['timings'] = {
            'get_mask (total)': get_mask_time,
            'commit': commit_time,
            'get_mask (special_map)': special_map_time
        }
        token_info['log'] = token_log
        
        if 'start_byte' in token_info:
            tokens_data.append(token_info)

    # Generate HTML
    html_parts = []
    last_pos = 0
    for token in sorted(tokens_data, key=lambda t: t['start_byte']):
        html_parts.append(html.escape(full_text[last_pos:token['start_byte']]))
        
        color = generate_color(token['timings']['get_mask (special_map)'])
        token_text = html.escape(full_text[token['start_byte']:token['end_byte']])
        data_log = html.escape(json.dumps(token))
        
        html_parts.append(f'<span class="token" style="background-color: {color};" data-tokeninfo=\'{data_log}\'>{token_text}</span>')
        last_pos = token['end_byte']
    
    html_parts.append(html.escape(full_text[last_pos:]))
    code_html = "".join(html_parts)

    output_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Performance Visualization</title>
    <style>
        body {{ font-family: sans-serif; display: flex; height: 100vh; margin: 0; }}
        #code-container {{ flex: 1; overflow: auto; padding: 1em; border-right: 1px solid #ccc; white-space: pre-wrap; font-family: monospace; line-height: 1.4; }}
        #details-container {{ flex: 1; overflow: auto; padding: 1em; }}
        .token {{ cursor: pointer; border-radius: 3px; }}
        .token:hover {{ outline: 1px solid blue; }}
        .token.locked {{ outline: 2px solid blue; }}
        #details-container h3 {{ margin-top: 0; }}
        #details-content pre {{ background-color: #f4f4f4; padding: 1em; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; }}
    </style>
</head>
<body>
    <div id="code-container"><pre>{code_html}</pre></div>
    <div id="details-container">
        <h3>Token Details</h3>
        <p>Hover over or click a token on the left to see its details.</p>
        <div id="details-content"></div>
    </div>

    <script>
        let lockedToken = null;

        document.getElementById('code-container').addEventListener('mouseover', (e) => {{
            if (e.target.classList.contains('token') && lockedToken === null) {{
                updateDetails(e.target);
            }}
        }});

        document.getElementById('code-container').addEventListener('click', (e) => {{
            if (e.target.classList.contains('token')) {{
                if (lockedToken) {{
                    lockedToken.classList.remove('locked');
                }}
                if (lockedToken === e.target) {{
                    lockedToken = null;
                    clearDetails();
                }} else {{
                    lockedToken = e.target;
                    lockedToken.classList.add('locked');
                    updateDetails(lockedToken);
                }}
            }}
        }});

        function updateDetails(tokenElement) {{
            const info = JSON.parse(tokenElement.dataset.tokeninfo);
            const detailsDiv = document.getElementById('details-content');
            
            let timingsHtml = '<ul>';
            for (const [key, value] of Object.entries(info.timings)) {{
                timingsHtml += `<li><strong>${{key}}:</strong> ${{value !== null ? value.toFixed(3) + ' ms' : 'N/A'}}</li>`;
            }}
            timingsHtml += '</ul>';

            detailsDiv.innerHTML = `
                <h4>Token Info</h4>
                <p><strong>Text:</strong> <code>${{html.escape(info.str)}}</code></p>
                <p><strong>ID:</strong> ${{info.id}}</p>
                <p><strong>Bytes:</strong> [${{info.start_byte}}, ${{info.end_byte}})</p>
                <h4>Timings</h4>
                ${{timingsHtml}}
                <h4>Full Log</h4>
                <pre>${{html.escape(info.log)}}</pre>
            `;
        }}
        
        function clearDetails() {{
            document.getElementById('details-content').innerHTML = '<p>Hover over or click a token on the left to see its details.</p>';
        }}

        /* -------------------------------------------------------------------
           Simple HTML-escaper for the small snippets we put into the details
           pane. We can’t rely on the browser’s implicit escaping because we’re
           building the markup with template-literals.
        ------------------------------------------------------------------- */
        const entityMap = {{
            '&':  '&amp;',
            '<':  '&lt;',
            '>':  '&gt;',
            '"':  '&quot;',
            "'":  '&#39;',
            '/':  '&#x2F;',
            '`':  '&#x60;',
            '=':  '&#x3D;'
        }};

        function escapeHtml(string) {{
            return String(string).replace(/[&<>"'`=\/]/g, s => entityMap[s]);
        }}

        /* expose as window.html.escape so that updateDetails() can use it */
        window.html = {{ escape: escapeHtml }};
    </script>
</body>
</html>
    """
    
    output_path.write_text(output_html, encoding='utf-8')
    print(f"Generated {output_path.resolve()}")

if __name__ == "__main__":
    main()