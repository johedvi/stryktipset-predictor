"""
Generate HTML viewer from Stryktipset coupon files
Automatically creates a beautiful web interface to view your strategies
"""

import os
import re
from datetime import datetime
from pathlib import Path


def parse_coupon_file(filepath):
    """Parse a coupon text file and extract all information"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract metadata
    week_match = re.search(r'Week:\s+(\d+),\s+(\d+)', content)
    strategy_match = re.search(r'Strategy:\s+#(\d+)', content)
    cost_match = re.search(r'Cost:\s+(\d+)\s+SEK', content)
    singles_match = re.search(r'Singles:\s+(\d+)', content)
    doubles_match = re.search(r'Doubles:\s+(\d+)', content)
    triples_match = re.search(r'Triples:\s+(\d+)', content)
    
    week = int(week_match.group(1)) if week_match else 0
    year = int(week_match.group(2)) if week_match else 0
    strategy_num = int(strategy_match.group(1)) if strategy_match else 0
    cost = int(cost_match.group(1)) if cost_match else 0
    singles = int(singles_match.group(1)) if singles_match else 0
    doubles = int(doubles_match.group(1)) if doubles_match else 0
    triples = int(triples_match.group(1)) if triples_match else 0
    
    # Extract probability table
    matches = []
    table_pattern = r'\s*(\d+)\s+\|\s+(.+?)\s+\|\s+(\S+)\s+\|\s+([\d.]+)%\s+\|\s+([\d.]+)%\s+\|\s+([\d.]+)%'
    
    for match in re.finditer(table_pattern, content):
        num = int(match.group(1))
        teams = match.group(2).strip()
        signs = match.group(3).strip()
        prob_1 = float(match.group(4))
        prob_x = float(match.group(5))
        prob_2 = float(match.group(6))
        
        matches.append({
            'num': num,
            'teams': teams,
            'signs': signs,
            'probs': {
                '1': round(prob_1),
                'X': round(prob_x),
                '2': round(prob_2)
            }
        })
    
    # Calculate score (you can adjust this formula)
    score = cost * 0.5 + singles * 10 + doubles * 5
    
    return {
        'num': strategy_num,
        'week': week,
        'year': year,
        'cost': cost,
        'singles': singles,
        'doubles': doubles,
        'triples': triples,
        'score': score,
        'matches': matches
    }


def find_latest_coupons(folder='coupons', max_strategies=3):
    """Find the most recent coupon files"""
    coupon_files = []
    
    for file in Path(folder).glob('stryktipset_week*.txt'):
        # Extract timestamp from filename
        timestamp_match = re.search(r'_(\d{8}_\d{6})\.txt$', str(file))
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            coupon_files.append((timestamp, str(file)))
    
    # Sort by timestamp (most recent first) and group by timestamp
    coupon_files.sort(reverse=True)
    
    if not coupon_files:
        return []
    
    # Get files from the most recent run (same timestamp base)
    latest_timestamp = coupon_files[0][0]
    latest_files = [f for t, f in coupon_files if t == latest_timestamp]
    
    return latest_files[:max_strategies]


def generate_html(strategies_data):
    """Generate the complete HTML file"""
    
    # Get week and year from first strategy
    week = strategies_data[0]['week'] if strategies_data else 42
    year = strategies_data[0]['year'] if strategies_data else 2025
    
    # Convert strategies to JSON format for JavaScript
    strategies_json = []
    for s in strategies_data:
        strategies_json.append({
            'num': s['num'],
            'cost': s['cost'],
            'singles': s['singles'],
            'doubles': s['doubles'],
            'triples': s['triples'],
            'score': s['score'],
            'matches': s['matches']
        })
    
    import json
    strategies_json_str = json.dumps(strategies_json, indent=12)
    
    html_template = f'''<!DOCTYPE html>
<html lang="sv">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stryktipset Vecka {week}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}

        h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        .week-info {{
            font-size: 1.2rem;
            opacity: 0.9;
        }}

        .generated-time {{
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 5px;
        }}

        .strategies-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .strategy-card {{
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            transition: transform 0.3s ease;
        }}

        .strategy-card:hover {{
            transform: translateY(-5px);
        }}

        .strategy-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}

        .strategy-header h2 {{
            font-size: 1.5rem;
            margin-bottom: 10px;
        }}

        .strategy-cost {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 10px;
        }}

        .strategy-stats {{
            display: flex;
            justify-content: space-around;
            font-size: 0.9rem;
        }}

        .stat-item {{
            text-align: center;
        }}

        .stat-value {{
            font-size: 1.5rem;
            font-weight: bold;
        }}

        .matches-container {{
            padding: 20px;
            max-height: 600px;
            overflow-y: auto;
        }}

        .match-row {{
            display: flex;
            align-items: center;
            padding: 12px;
            border-bottom: 1px solid #eee;
            transition: background 0.2s;
        }}

        .match-row:hover {{
            background: #f5f5f5;
        }}

        .match-num {{
            width: 30px;
            font-weight: bold;
            color: #667eea;
        }}

        .match-sign {{
            width: 50px;
            text-align: center;
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 5px;
            background: #667eea;
            color: white;
            margin-right: 15px;
            font-size: 0.9rem;
        }}

        .match-sign.single {{
            background: #10b981;
        }}

        .match-sign.double {{
            background: #f59e0b;
        }}

        .match-sign.triple {{
            background: #ef4444;
        }}

        .match-teams {{
            flex: 1;
            font-size: 0.9rem;
        }}

        .match-probs {{
            display: flex;
            gap: 8px;
            font-size: 0.85rem;
            color: #666;
        }}

        .prob-item {{
            padding: 3px 8px;
            border-radius: 3px;
            background: #f0f0f0;
        }}

        .prob-item.selected {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}

        .comparison-table {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}

        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}

        tr:hover {{
            background: #f5f5f5;
        }}

        .best-value {{
            background: #10b981;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.8rem;
        }}

        .print-btn {{
            background: white;
            color: #667eea;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            margin-top: 10px;
            transition: transform 0.2s;
        }}

        .print-btn:hover {{
            transform: scale(1.05);
        }}

        @media (max-width: 768px) {{
            .strategies-grid {{
                grid-template-columns: 1fr;
            }}
            
            h1 {{
                font-size: 1.8rem;
            }}

            .match-probs {{
                flex-direction: column;
                gap: 4px;
            }}
        }}

        @media print {{
            body {{
                background: white;
            }}
            
            .strategy-card {{
                break-inside: avoid;
                page-break-inside: avoid;
            }}

            .print-btn {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⚽ Stryktipset Kupongvisare</h1>
            <div class="week-info">Vecka {week}, {year}</div>
            <div class="generated-time">Genererad: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <button class="print-btn" onclick="window.print()">🖨️ Skriv ut</button>
        </header>

        <div class="strategies-grid" id="strategiesGrid">
            <!-- Strategies will be inserted here -->
        </div>

        <div class="comparison-table">
            <h2 style="margin-bottom: 20px; color: #667eea;">📊 Jämförelse av Strategier</h2>
            <table id="comparisonTable">
                <thead>
                    <tr>
                        <th>Strategi</th>
                        <th>Kostnad</th>
                        <th>Ettor</th>
                        <th>Tvåor</th>
                        <th>Treor</th>
                        <th>Antal rader</th>
                    </tr>
                </thead>
                <tbody id="comparisonBody">
                    <!-- Comparison rows will be inserted here -->
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const couponsData = {{
            week: {week},
            year: {year},
            strategies: {strategies_json_str}
        }};

        function getSignClass(signs) {{
            if (signs.length === 1) return 'single';
            if (signs.length === 2) return 'double';
            return 'triple';
        }}

        function isSignSelected(sign, signs) {{
            return signs.includes(sign);
        }}

        function renderStrategy(strategy) {{
            const matchesHtml = strategy.matches.map(match => `
                <div class="match-row">
                    <div class="match-num">${{match.num}}</div>
                    <div class="match-sign ${{getSignClass(match.signs)}}">${{match.signs}}</div>
                    <div class="match-teams">${{match.teams}}</div>
                    <div class="match-probs">
                        <span class="prob-item ${{isSignSelected('1', match.signs) ? 'selected' : ''}}">1: ${{match.probs['1']}}%</span>
                        <span class="prob-item ${{isSignSelected('X', match.signs) ? 'selected' : ''}}">X: ${{match.probs['X']}}%</span>
                        <span class="prob-item ${{isSignSelected('2', match.signs) ? 'selected' : ''}}">2: ${{match.probs['2']}}%</span>
                    </div>
                </div>
            `).join('');

            return `
                <div class="strategy-card">
                    <div class="strategy-header">
                        <h2>Strategi #${{strategy.num}}</h2>
                        <div class="strategy-cost">${{strategy.cost}} SEK</div>
                        <div class="strategy-stats">
                            <div class="stat-item">
                                <div class="stat-value">${{strategy.singles}}</div>
                                <div>Ettor</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">${{strategy.doubles}}</div>
                                <div>Tvåor</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">${{strategy.triples}}</div>
                                <div>Treor</div>
                            </div>
                        </div>
                    </div>
                    <div class="matches-container">
                        ${{matchesHtml}}
                    </div>
                </div>
            `;
        }}

        function renderComparison() {{
            const bestCost = Math.min(...couponsData.strategies.map(s => s.cost));

            return couponsData.strategies.map(strategy => `
                <tr>
                    <td><strong>Strategi #${{strategy.num}}</strong></td>
                    <td>${{strategy.cost}} SEK ${{strategy.cost === bestCost ? '<span class="best-value">Billigast</span>' : ''}}</td>
                    <td>${{strategy.singles}}</td>
                    <td>${{strategy.doubles}}</td>
                    <td>${{strategy.triples}}</td>
                    <td>${{strategy.cost}} rader</td>
                </tr>
            `).join('');
        }}

        // Render everything
        document.getElementById('strategiesGrid').innerHTML = couponsData.strategies.map(renderStrategy).join('');
        document.getElementById('comparisonBody').innerHTML = renderComparison();
    </script>
</body>
</html>'''
    
    return html_template


def main():
    """Main function to generate the HTML viewer"""
    print("🔍 Looking for coupon files...")
    
    coupon_files = find_latest_coupons()
    
    if not coupon_files:
        print("❌ No coupon files found in 'coupons/' folder")
        print("💡 Run your prediction script first to generate coupons")
        return
    
    print(f"✓ Found {len(coupon_files)} coupon files")
    
    # Parse all coupon files
    strategies = []
    for filepath in coupon_files:
        print(f"  📄 Parsing: {filepath}")
        try:
            strategy_data = parse_coupon_file(filepath)
            strategies.append(strategy_data)
        except Exception as e:
            print(f"  ⚠️  Error parsing {filepath}: {e}")
    
    if not strategies:
        print("❌ Could not parse any coupon files")
        return
    
    # Sort by strategy number
    strategies.sort(key=lambda x: x['num'])
    
    print(f"\n✓ Successfully parsed {len(strategies)} strategies")
    
    # Generate HTML
    print("\n🎨 Generating HTML viewer...")
    html_content = generate_html(strategies)
    
    # Save HTML file
    output_file = 'stryktipset_viewer.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML viewer generated: {output_file}")
    print(f"\n🌐 Open {output_file} in your browser to view your coupons!")
    print(f"📍 Full path: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()