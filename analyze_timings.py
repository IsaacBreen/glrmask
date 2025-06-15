import re
import html
import math
import ast


def parse_time(s: str) -> float:
    """Parses time string like '1.234ms', '567µs', '2.3s' into seconds."""
    s = s.strip()
    if s.endswith('ms'):
        return float(s[:-2]) / 1000.0
    elif s.endswith('µs') or s.endswith('us'):
        return float(s[:-2]) / 1000000.0
    elif s.endswith('s'):
        return float(s[:-1])
    return 0.0


def get_color_for_time(time_sec: float, max_time: float) -> str:
    """Generates a color from green to red based on time, using a log scale."""
    if max_time == 0 or time_sec <= 0:
        return "#d0f0c0"  # A light green for zero/negligible time

    # Use a logarithmic scale for better color distribution.
    # Add a small epsilon to avoid log(0) issues.
    log_time = math.log(time_sec + 1e-9)
    log_max_time = math.log(max_time + 1e-9)

    # Normalize the log-scaled time.
    # To avoid division by zero if max_time is very small, we need a sensible baseline.
    # Let's assume the "coldest" color corresponds to a very small time.
    # A reasonable range for log-time might be from log(1µs) to log(max_time).
    log_min_time_ref = math.log(1e-6) # 1 microsecond
    
    # Ensure our scale starts from at least our reference minimum.
    effective_log_min = min(log_min_time_ref, log_time)

    if log_max_time <= effective_log_min:
        ratio = 0.0
    else:
        ratio = (log_time - effective_log_min) / (log_max_time - effective_log_min)
    
    ratio = max(0, min(1, ratio))  # Clamp to [0, 1]

    # Linear interpolation between green (120) and red (0) in HSL color space.
    hue = 120 * (1 - ratio)
    return f"hsl({hue}, 80%, 70%)"


def main():
    log_file = '.temp2'
    source_file = 'src/example_code2.py'
    output_file = 'timings_visualization.html'

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found. Run your test and redirect stdout.")
        return

    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except FileNotFoundError:
        print(f"Error: Source file '{source_file}' not found.")
        return

    chunks = log_content.split('\n---- TOKEN START ----\n')

    token_data = []
    for chunk in chunks[1:]:  # Skip the part before the first delimiter
        if not chunk.strip():
            continue

        token_match = re.search(r"token_str: (.+)", chunk)
        if not token_match:
            continue

        token_repr = token_match.group(1)
        token_str = ast.literal_eval(token_repr)

        # Time for coloring is from special_map
        coloring_time = 0
        time_match = re.search(r'after special_map:\s+([\d.]+\w?s)', chunk)
        if time_match:
            coloring_time = parse_time(time_match.group(1))

        # Extract other timings for tooltip
        get_mask_time = 0
        get_mask_match = re.search(r'get_mask took:\s+([\d.]+\w?s)', chunk)
        if get_mask_match:
            get_mask_time = parse_time(get_mask_match.group(1))

        commit_time = 0
        commit_match = re.search(r'commit LLMTokenID\(\d+\) took:\s+([\d.]+\w?s)', chunk)
        if commit_match:
            commit_time = parse_time(commit_match.group(1))

        token_data.append({
            'token': token_str,
            'coloring_time': coloring_time,
            'get_mask_time': get_mask_time,
            'commit_time': commit_time,
            'log': chunk
        })

    # The first entry is for "<initial>". Its time belongs to the first real token.
    if token_data and token_data[0]['token'] == '<initial>':
        initial_data = token_data.pop(0)
        if token_data:
            # The get_mask time from the initial step belongs to the first real token
            token_data[0]['coloring_time'] = initial_data['coloring_time']
            token_data[0]['get_mask_time'] = initial_data['get_mask_time']
            # Prepend the initial log to the first real token's log for context
            token_data[0]['log'] = initial_data['log'] + '\n---- TOKEN START ----\n' + token_data[0]['log']

    max_time = max(d.get('coloring_time', 0) for d in token_data) if token_data else 0

    # Generate HTML from source code and token data
    html_parts = []
    current_pos = 0
    for data in token_data:
        token = data['token']

        try:
            token_pos = source_code.index(token, current_pos)
        except ValueError:
            print(f"Warning: Token {repr(token)} not found in source after position {current_pos}.")
            continue

        # Add preceding text (whitespace, etc.)
        if token_pos > current_pos:
            html_parts.append(html.escape(source_code[current_pos:token_pos]))

        color = get_color_for_time(data['coloring_time'], max_time)
        escaped_log = html.escape(data['log'])
        
        # Build a more detailed title
        title_parts = [
            f"{html.escape(repr(token))}",
            f"special_map: {data['coloring_time']*1000:.3f} ms",
            f"get_mask total: {data['get_mask_time']*1000:.3f} ms",
            f"commit: {data['commit_time']*1000:.3f} ms",
            "\n--- LOG ---",
            escaped_log
        ]
        title = "\n".join(title_parts)

        span = (f'<span class="token" style="background-color: {color};" title="{title}">'
                f'{html.escape(token)}</span>')
        html_parts.append(span)

        current_pos = token_pos + len(token)

    if current_pos < len(source_code):
        html_parts.append(html.escape(source_code[current_pos:]))

    html_body = "".join(html_parts)
    html_template = f"""<!DOCTYPE html><html><head><title>Timings Visualization</title><style>
    body {{ font-family: monospace; white-space: pre; background-color: #fdf6e3; color: #657b83; }}
    .token {{ cursor: help; border-radius: 3px; }}
    h1, p {{ font-family: sans-serif; }}
</style></head><body>
<h1>Performance for {html.escape(source_file)}</h1>
<p>Hover over tokens to see timing and log details. Colors are based on 'special_map' time and are on a logarithmic scale from green (fast) to red (slow).</p><hr>
{html_body}</body></html>"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"Visualization saved to '{output_file}'")

if __name__ == '__main__':
    main()
