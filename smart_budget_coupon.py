"""
Optimized Stryktipset Coupon Generator
Creates optimal coupons up to 192 SEK budget using intelligent distribution
Formula: Cost = 2^(doubles) × 3^(triples)
"""

import os
from datetime import datetime


def get_week_number():
    """Get current ISO week number"""
    return datetime.now().isocalendar()[1]


def get_year():
    """Get current year"""
    return datetime.now().year


def create_optimal_192_coupon(predictions, max_budget=192, output_folder="coupons", week_number=None, num_strategies=3):
    """
    Create the best possible coupon within the specified budget
    
    Args:
        predictions: List of 13 match predictions with probabilities
        max_budget: Maximum budget in SEK (default: 192)
        output_folder: Folder to save coupon files (default: "coupons")
        week_number: Stryktipset week number (default: current ISO week)
        num_strategies: Number of different strategies to generate (default: 3)
    
    Returns:
        List of optimized coupons (or single coupon if num_strategies=1)
    """
    # Get week number if not provided
    if week_number is None:
        week_number = get_week_number()
    
    year = get_year()
    
    print("\n" + "="*100)
    print(f"CREATING {num_strategies} OPTIMAL STRATEGIES - WEEK {week_number}, {year} (MAX BUDGET: {max_budget} SEK)")
    print("="*100)
    print("\nFormula: Cost = 2^(doubles) × 3^(triples)")
    print(f"Target: Use as much of {max_budget} SEK as possible (but not more!)")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\n✓ Created folder: {output_folder}/")
    
    # Validate input
    if len(predictions) != 13:
        raise ValueError(f"Expected 13 predictions, got {len(predictions)}")
    
    # Step 1: Analyze all matches and calculate confidence metrics
    match_analysis = []
    for i, pred in enumerate(predictions, 1):
        probs = pred.get('probabilities', {'H': 0.33, 'D': 0.34, 'A': 0.33})
        
        h_prob = float(probs.get('H', 0.33))
        d_prob = float(probs.get('D', 0.34))
        a_prob = float(probs.get('A', 0.33))
        
        # Sort probabilities to get confidence metrics
        sorted_probs = sorted([h_prob, d_prob, a_prob], reverse=True)
        highest = sorted_probs[0]
        second = sorted_probs[1]
        lowest = sorted_probs[2]
        
        # Key confidence metrics
        confidence_gap = highest - second  # How clear is the winner?
        spread = highest - lowest  # How spread out are all outcomes?
        top2_sum = highest + second  # Combined confidence in top 2
        
        match_analysis.append({
            'match_num': i,
            'match': f"{pred['home_team']} vs {pred['away_team']}",
            'highest_prob': highest,
            'confidence_gap': confidence_gap,
            'spread': spread,
            'top2_sum': top2_sum,
            'probabilities': probs,
            'h_prob': h_prob,
            'd_prob': d_prob,
            'a_prob': a_prob,
            'pred': pred
        })
    
    # Step 2: Classify matches by confidence type
    certain_matches = []
    uncertain_matches = []
    low_draw_matches = []
    
    for m in match_analysis:
        if m['spread'] < 0.10:
            uncertain_matches.append(m)
        elif m['d_prob'] < 0.20 and abs(m['h_prob'] - m['a_prob']) < 0.15:
            low_draw_matches.append(m)
        else:
            certain_matches.append(m)
    
    certain_matches.sort(key=lambda x: x['confidence_gap'], reverse=True)
    
    print("\n" + "-"*100)
    print("MATCH CLASSIFICATION:")
    print("-"*100)
    print(f"\n✓ CERTAIN MATCHES ({len(certain_matches)}):")
    for m in certain_matches:
        print(f"  {m['match_num']:2d}. {m['match']:45s} Gap: {m['confidence_gap']:.3f} | {m['h_prob']:.1%}/{m['d_prob']:.1%}/{m['a_prob']:.1%}")
    
    if low_draw_matches:
        print(f"\n🎯 LOW DRAW MATCHES ({len(low_draw_matches)}) - Good for 12:")
        for m in low_draw_matches:
            print(f"  {m['match_num']:2d}. {m['match']:45s} Draw: {m['d_prob']:.1%} | {m['h_prob']:.1%}/{m['d_prob']:.1%}/{m['a_prob']:.1%}")
    
    if uncertain_matches:
        print(f"\n⚠ UNCERTAIN MATCHES ({len(uncertain_matches)}) - Should be triples:")
        for m in uncertain_matches:
            print(f"  {m['match_num']:2d}. {m['match']:45s} Spread: {m['spread']:.3f} | {m['h_prob']:.1%}/{m['d_prob']:.1%}/{m['a_prob']:.1%}")
    
    match_analysis = certain_matches + low_draw_matches + uncertain_matches
    
    # Step 3: Find optimal distributions
    print("\n" + "-"*100)
    print("FINDING OPTIMAL DISTRIBUTIONS:")
    print("-"*100)
    
    possible_combinations = []
    for triples in range(14):
        for doubles in range(14 - triples):
            singles = 13 - doubles - triples
            if singles >= 0:
                cost = (2 ** doubles) * (3 ** triples)
                if cost <= max_budget:
                    possible_combinations.append({
                        'singles': singles,
                        'doubles': doubles,
                        'triples': triples,
                        'cost': cost
                    })
    
    possible_combinations.sort(key=lambda x: x['cost'], reverse=True)
    
    print(f"\nTop 10 combinations ≤ {max_budget} SEK:")
    for combo in possible_combinations[:10]:
        print(f"  • {combo['cost']:3d} SEK: {combo['singles']}S + {combo['doubles']}D + {combo['triples']}T (2^{combo['doubles']} × 3^{combo['triples']})")
    
    # Step 4: Score and select top N strategies
    scored_combinations = []
    min_required_triples = len(uncertain_matches)
    
    for combo in possible_combinations:
        if combo['triples'] < min_required_triples:
            continue
        
        score = 0
        certain_idx = 0
        
        for i in range(combo['singles']):
            if certain_idx < len(certain_matches):
                score += certain_matches[certain_idx]['confidence_gap'] * 100
                certain_idx += 1
            else:
                score -= 50
        
        for i in range(combo['doubles']):
            if certain_idx < len(certain_matches):
                m = certain_matches[certain_idx]
                score += m['top2_sum'] * 50
                if 0.08 < m['confidence_gap'] < 0.25:
                    score += 15
                certain_idx += 1
            else:
                score -= 50
        
        extra_triples = combo['triples'] - min_required_triples
        for i in range(extra_triples):
            if certain_idx < len(certain_matches):
                m = certain_matches[certain_idx]
                if m['confidence_gap'] > 0.25:
                    score -= 20
                elif m['confidence_gap'] > 0.15:
                    score -= 10
                certain_idx += 1
        
        cost_penalty = (max_budget - combo['cost']) * 3
        score -= cost_penalty
        
        scored_combinations.append({'combo': combo, 'score': score})
    
    scored_combinations.sort(key=lambda x: x['score'], reverse=True)
    
    # Select diverse strategies
    selected_strategies = []
    seen_signatures = set()
    
    for item in scored_combinations:
        combo = item['combo']
        cost_bucket = (combo['cost'] // 32) * 32
        signature = f"{cost_bucket}_{combo['singles']}_{combo['doubles']}_{combo['triples']}"
        
        if signature not in seen_signatures:
            selected_strategies.append(item)
            seen_signatures.add(signature)
        
        if len(selected_strategies) >= num_strategies:
            break
    
    print(f"\n✓ Selected {len(selected_strategies)} different strategies:")
    for i, item in enumerate(selected_strategies, 1):
        combo = item['combo']
        print(f"  {i}. {combo['singles']}S + {combo['doubles']}D + {combo['triples']}T = {combo['cost']} SEK (Score: {item['score']:.1f})")
    
    # Generate coupons for each strategy
    all_coupons = []
    
    for strategy_num, item in enumerate(selected_strategies, 1):
        best_combo = item['combo']
        best_score = item['score']
        
        print("\n" + "="*100)
        print(f"STRATEGY #{strategy_num}: {best_combo['singles']}S + {best_combo['doubles']}D + {best_combo['triples']}T = {best_combo['cost']} SEK")
        print("="*100)
        
        coupon = build_coupon(best_combo, certain_matches, uncertain_matches, match_analysis)
        
        singles_count = sum(1 for c in coupon if len(c['signs']) == 1)
        doubles_count = sum(1 for c in coupon if len(c['signs']) == 2)
        triples_count = sum(1 for c in coupon if len(c['signs']) == 3)
        actual_cost = (2 ** doubles_count) * (3 ** triples_count)
        
        print(f"\n✅ Strategy #{strategy_num} Complete: {singles_count}S + {doubles_count}D + {triples_count}T = {actual_cost} SEK")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stryktipset_week{week_number}_{year}_strategy{strategy_num}_{actual_cost}SEK_{timestamp}.txt"
        filepath = os.path.join(output_folder, filename)
        
        save_coupon_to_file(coupon, filepath, actual_cost, max_budget, singles_count, doubles_count, triples_count, week_number, year, strategy_num)
        
        print(f"💾 Saved to: {filename}")
        
        all_coupons.append({
            'strategy_num': strategy_num,
            'coupon': coupon,
            'cost': actual_cost,
            'singles': singles_count,
            'doubles': doubles_count,
            'triples': triples_count,
            'score': best_score
        })
    
    print("\n" + "="*100)
    print(f"ALL {len(all_coupons)} STRATEGIES GENERATED!")
    print("="*100)
    for strategy in all_coupons:
        print(f"  Strategy #{strategy['strategy_num']}: {strategy['singles']}S + {strategy['doubles']}D + {strategy['triples']}T = {strategy['cost']} SEK")
    print("="*100)
    
    return all_coupons if num_strategies > 1 else all_coupons[0]['coupon']


def build_coupon(best_combo, certain_matches, uncertain_matches, match_analysis):
    """Build a coupon based on the selected strategy"""
    coupon = []
    certain_idx = 0
    
    # Singles
    for i in range(best_combo['singles']):
        if certain_idx >= len(certain_matches):
            break
        m = certain_matches[certain_idx]
        outcomes = [('1', m['h_prob']), ('X', m['d_prob']), ('2', m['a_prob'])]
        best_outcome = max(outcomes, key=lambda x: x[1])
        
        coupon.append({
            'match_num': m['match_num'],
            'match': m['match'],
            'signs': best_outcome[0],
            'strategy': 'single',
            'confidence': best_outcome[1],
            'probabilities': m['probabilities']
        })
        certain_idx += 1
    
    # Doubles
    for i in range(best_combo['doubles']):
        if certain_idx >= len(certain_matches):
            break
        m = certain_matches[certain_idx]
        
        if m['d_prob'] < 0.20 and abs(m['h_prob'] - m['a_prob']) < 0.15:
            coupon.append({
                'match_num': m['match_num'],
                'match': m['match'],
                'signs': '12',
                'strategy': 'double',
                'confidence': m['h_prob'] + m['a_prob'],
                'probabilities': m['probabilities']
            })
        else:
            outcomes = [('1', m['h_prob']), ('X', m['d_prob']), ('2', m['a_prob'])]
            outcomes.sort(key=lambda x: x[1], reverse=True)
            double_signs = ''.join(sorted([outcomes[0][0], outcomes[1][0]]))
            
            coupon.append({
                'match_num': m['match_num'],
                'match': m['match'],
                'signs': double_signs,
                'strategy': 'double',
                'confidence': outcomes[0][1] + outcomes[1][1],
                'probabilities': m['probabilities']
            })
        certain_idx += 1
    
    # Triples
    for m in uncertain_matches:
        coupon.append({
            'match_num': m['match_num'],
            'match': m['match'],
            'signs': '1X2',
            'strategy': 'triple_uncertain',
            'confidence': 1.0,
            'probabilities': m['probabilities']
        })
    
    remaining_triples = best_combo['triples'] - len(uncertain_matches)
    for i in range(remaining_triples):
        if certain_idx >= len(certain_matches):
            break
        m = certain_matches[certain_idx]
        coupon.append({
            'match_num': m['match_num'],
            'match': m['match'],
            'signs': '1X2',
            'strategy': 'triple_extra',
            'confidence': 1.0,
            'probabilities': m['probabilities']
        })
        certain_idx += 1
    
    coupon.sort(key=lambda x: x['match_num'])
    return coupon


def save_coupon_to_file(coupon, filepath, cost, max_budget, singles, doubles, triples, week_number, year, strategy_num=1):
    """Save the coupon to a text file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"STRYKTIPSET COUPON - STRATEGY #{strategy_num}\n")
        f.write("="*100 + "\n")
        f.write(f"\n📅 WEEK {week_number}, {year}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Strategy: #{strategy_num}\n")
        f.write(f"Cost: {cost} SEK (Budget: {max_budget} SEK)\n")
        f.write(f"Distribution: {singles} Singles + {doubles} Doubles + {triples} Triples\n")
        f.write(f"Formula: 2^{doubles} × 3^{triples} = {cost} SEK\n")
        f.write("\n" + "="*100 + "\n")
        f.write("MATCHES:\n")
        f.write("="*100 + "\n\n")
        
        for match in coupon:
            signs = match['signs']
            probs = match['probabilities']
            f.write(f"Match {match['match_num']:2d}: [{signs:3s}]  {match['match']:50s}\n")
            
            coverage = []
            if '1' in signs:
                coverage.append(f"  1 (Home):  {probs['H']:.1%}")
            if 'X' in signs:
                coverage.append(f"  X (Draw):  {probs['D']:.1%}")
            if '2' in signs:
                coverage.append(f"  2 (Away):  {probs['A']:.1%}")
            
            for line in coverage:
                f.write(f"          {line}\n")
            
            strategy_notes = {
                'single': 'High confidence - single pick',
                'double': 'Medium confidence - double coverage',
                'triple_uncertain': 'Too close to call - full coverage',
                'triple_extra': 'Low confidence - full coverage'
            }
            f.write(f"          Strategy: {strategy_notes.get(match['strategy'], match['strategy'])}\n\n")
        
        f.write("="*100 + "\n")
        f.write("QUICK REFERENCE (for submission):\n")
        f.write("="*100 + "\n\n")
        
        for match in coupon:
            f.write(f" {match['match_num']:2d}. [{match['signs']:3s}]\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("PROBABILITY OVERVIEW TABLE:\n")
        f.write("="*100 + "\n\n")
        f.write("Match | Team                                      | Sign | 1 (Home) | X (Draw) | 2 (Away)\n")
        f.write("-"*100 + "\n")
        
        for match in coupon:
            probs = match['probabilities']
            match_name = match['match'][:42]
            f.write(f" {match['match_num']:2d}   | {match_name:42s} | {match['signs']:3s}  | {probs['H']:6.1%}   | {probs['D']:6.1%}   | {probs['A']:6.1%}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("SUMMARY:\n")
        f.write("="*100 + "\n")
        f.write(f"Strategy:       #{strategy_num}\n")
        f.write(f"Week:           {week_number}, {year}\n")
        f.write(f"Total matches:  13\n")
        f.write(f"Singles:        {singles}\n")
        f.write(f"Doubles:        {doubles}\n")
        f.write(f"Triples:        {triples}\n")
        f.write(f"Total rows:     {cost}\n")
        f.write(f"Cost:           {cost} SEK\n")
        f.write("="*100 + "\n")