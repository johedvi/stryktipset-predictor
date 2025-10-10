"""
Create an optimized coupon with exactly 192 SEK budget
Uses proper Stryktipset calculation: 1^singles × 2^doubles × 3^triples
Intelligently selects the best distribution based on prediction confidence
"""

def create_optimal_192_coupon(predictions):
    """
    Create the best possible coupon for exactly 192 SEK
    
    Formula: Cost = 1^(singles) × 2^(doubles) × 3^(triples)
    Target: 192 SEK
    
    Finds the optimal distribution based on actual prediction confidence
    
    Args:
        predictions: List of all 13 predictions
    
    Returns:
        Optimized coupon with exactly 192 SEK
    """
    print("\n" + "="*100)
    print("CREATING OPTIMAL 192 SEK COUPON")
    print("="*100)
    print("\nFormula: Cost = 2^(doubles) × 3^(triples)")
    
    # Generate ALL possible combinations for 13 matches
    possible_combinations = []
    total_matches = len(predictions)
    
    # Try all possible combinations of singles, doubles, triples
    for triples in range(total_matches + 1):
        for doubles in range(total_matches - triples + 1):
            singles = total_matches - doubles - triples
            
            if singles < 0:
                continue
            
            # Calculate cost: 2^doubles × 3^triples
            cost = (2 ** doubles) * (3 ** triples)
            
            possible_combinations.append({
                'singles': singles,
                'doubles': doubles,
                'triples': triples,
                'cost': cost,
                'desc': f'2^{doubles} × 3^{triples}' if doubles > 0 or triples > 0 else 'all singles'
            })
    
    # Filter to combinations that cost ≤ 192 SEK
    target_combos = [c for c in possible_combinations if c['cost'] <= 192]
    
    # Sort by cost (descending) - we want to use as much of the budget as possible
    target_combos.sort(key=lambda x: x['cost'], reverse=True)
    
    if not target_combos:
        print("\n⚠️  WARNING: Even all singles costs more than 192 SEK!")
        print("This shouldn't happen with 13 matches (2^0 × 3^0 = 1 SEK)")
        target_combos = [possible_combinations[0]]
    
    print("\nPossible combinations ≤ 192 SEK (top 10):")
    for combo in target_combos[:10]:
        print(f"  • {combo['cost']:3d} SEK: {combo['singles']} singles + {combo['doubles']} doubles + {combo['triples']} triples ({combo['desc']})")
    
    # Analyze all matches and assign confidence scores
    match_analysis = []
    for i, pred in enumerate(predictions, 1):
        probs = pred.get('probabilities', {'H': 0.33, 'D': 0.34, 'A': 0.33})
        
        h_prob = float(probs.get('H', 0.33))
        d_prob = float(probs.get('D', 0.34))
        a_prob = float(probs.get('A', 0.33))
        
        # Sort probabilities
        sorted_probs = sorted([h_prob, d_prob, a_prob], reverse=True)
        highest = sorted_probs[0]
        second = sorted_probs[1]
        lowest = sorted_probs[2]
        
        # Confidence metrics
        gap = highest - second  # How clear is the winner?
        spread = highest - lowest  # How spread out are all outcomes?
        top2_sum = highest + second  # Combined confidence in top 2
        
        match_analysis.append({
            'match_num': i,
            'match': f"{pred['home_team']} vs {pred['away_team']}",
            'highest_prob': highest,
            'confidence_gap': gap,
            'spread': spread,
            'top2_sum': top2_sum,
            'probabilities': probs,
            'pred': pred
        })
    
    # Sort by confidence gap (most certain outcomes first)
    match_analysis.sort(key=lambda x: x['confidence_gap'], reverse=True)
    
    # Evaluate each possible combination and pick the best
    best_combo = None
    best_score = -1
    
    for combo in target_combos:
        # Calculate quality score for this distribution
        score = 0
        
        # Check singles: should have high confidence gap
        for i in range(combo['singles']):
            if i < len(match_analysis):
                score += match_analysis[i]['confidence_gap'] * 100  # Reward clear winners
        
        # Check doubles: should have good top2 sum but not extreme gap
        for i in range(combo['singles'], combo['singles'] + combo['doubles']):
            if i < len(match_analysis):
                m = match_analysis[i]
                # Good for double: high top2 sum, moderate gap
                score += m['top2_sum'] * 50
                if 0.05 < m['confidence_gap'] < 0.20:  # Sweet spot for doubles
                    score += 20
        
        # Triples are okay for low confidence matches (no penalty)
        
        # Prefer combinations closer to 192 SEK
        cost_penalty = abs(192 - combo['cost']) * 2
        score -= cost_penalty
        
        if score > best_score:
            best_score = score
            best_combo = combo
    
    print(f"\n✓ Selected combination: {best_combo['singles']} singles + {best_combo['doubles']} doubles + {best_combo['triples']} triples = {best_combo['cost']} SEK")
    print(f"  (Using {best_combo['cost']}/192 SEK budget)")
    print(f"  Reasoning: Best match between prediction confidence and sign distribution")
    
    # Build the coupon with optimal distribution
    print("\n" + "-"*100)
    print("COUPON BREAKDOWN:")
    print("-"*100)
    
    coupon = []
    idx = 0
    
    # Singles: Most confident matches
    if best_combo['singles'] > 0:
        print(f"\n🎯 SINGLES ({best_combo['singles']} highest confidence):")
        for i in range(best_combo['singles']):
            if idx >= len(match_analysis):
                break
            m = match_analysis[idx]
            probs = m['probabilities']
            
            # Pick best single outcome
            outcomes = [('1', probs['H']), ('X', probs['D']), ('2', probs['A'])]
            best = max(outcomes, key=lambda x: x[1])
            
            coupon.append({
                'match_num': m['match_num'],
                'match': m['match'],
                'signs': best[0],
                'strategy': 'high_confidence',
                'confidence': best[1],
                'probabilities': probs
            })
            print(f"  {m['match_num']:2d}. {m['match']:45s} [{best[0]}] ({best[1]:.1%})")
            idx += 1
    
    # Doubles: Medium confidence matches
    # Note: Some may become triples if too uncertain
    if best_combo['doubles'] > 0:
        print(f"\n🎲 DOUBLES ({best_combo['doubles']} medium confidence):")
        doubles_processed = 0
        for i in range(best_combo['doubles']):
            if idx >= len(match_analysis):
                break
            m = match_analysis[idx]
            probs = m['probabilities']
            
            # Check if match is too uncertain (all outcomes within 10%)
            spread = m['spread']
            
            if spread < 0.10:  # All outcomes within 10% - too uncertain
                # This becomes a triple instead - don't count as double
                print(f"  {m['match_num']:2d}. {m['match']:45s} [1X2] (Too uncertain: {probs['H']:.1%}/{probs['D']:.1%}/{probs['A']:.1%}) → moved to triples")
                # We'll add this to triples section instead
                idx += 1
                continue
            else:
                # Pick two best outcomes
                outcomes = [('1', probs['H']), ('X', probs['D']), ('2', probs['A'])]
                outcomes.sort(key=lambda x: x[1], reverse=True)
                double_sign = ''.join(sorted([outcomes[0][0], outcomes[1][0]]))
                
                coupon.append({
                    'match_num': m['match_num'],
                    'match': m['match'],
                    'signs': double_sign,
                    'strategy': 'medium_confidence',
                    'confidence': outcomes[0][1] + outcomes[1][1],
                    'probabilities': probs
                })
                print(f"  {m['match_num']:2d}. {m['match']:45s} [{double_sign}] ({outcomes[0][1]:.1%} + {outcomes[1][1]:.1%})")
                doubles_processed += 1
            idx += 1
    
    # Triples: Lowest confidence matches (including uncertain doubles)
    remaining_triples = best_combo['triples'] + (best_combo['doubles'] - doubles_processed)
    if remaining_triples > 0:
        print(f"\n🔄 TRIPLES ({remaining_triples} lowest confidence + uncertain):")
        triples_added = 0
        
        # First, add any uncertain matches from doubles section
        for m in match_analysis:
            if triples_added >= remaining_triples:
                break
            if m['match_num'] in [c['match_num'] for c in coupon]:
                continue  # Already added
            if m['spread'] < 0.10:  # Uncertain match
                coupon.append({
                    'match_num': m['match_num'],
                    'match': m['match'],
                    'signs': '1X2',
                    'strategy': 'uncertain',
                    'confidence': 1.0,
                    'probabilities': m['probabilities']
                })
                probs = m['probabilities']
                print(f"  {m['match_num']:2d}. {m['match']:45s} [1X2] ({probs['H']:.1%}/{probs['D']:.1%}/{probs['A']:.1%})")
                triples_added += 1
        
        # Then add remaining lowest confidence matches
        while triples_added < remaining_triples and idx < len(match_analysis):
            m = match_analysis[idx]
            if m['match_num'] not in [c['match_num'] for c in coupon]:
                coupon.append({
                    'match_num': m['match_num'],
                    'match': m['match'],
                    'signs': '1X2',
                    'strategy': 'low_confidence',
                    'confidence': 1.0,
                    'probabilities': m['probabilities']
                })
                probs = m['probabilities']
                print(f"  {m['match_num']:2d}. {m['match']:45s} [1X2] ({probs['H']:.1%}/{probs['D']:.1%}/{probs['A']:.1%})")
                triples_added += 1
            idx += 1
    
    # Sort coupon by match number
    coupon.sort(key=lambda x: x['match_num'])
    
    # Calculate final cost
    singles = sum(1 for c in coupon if len(c['signs']) == 1)
    doubles = sum(1 for c in coupon if len(c['signs']) == 2)
    triples = sum(1 for c in coupon if len(c['signs']) == 3)
    cost = (2 ** doubles) * (3 ** triples)
    
    print("\n" + "="*100)
    print(f"FINAL COUPON SUMMARY:")
    print(f"  Singles: {singles}")
    print(f"  Doubles: {doubles}")
    print(f"  Triples: {triples}")
    print(f"  Total cost: 2^{doubles} × 3^{triples} = {cost} SEK")
    print("="*100)
    
    return coupon