"""
Stryktipset signing strategy based on prediction confidence
"""

def get_sign_strategy(probabilities, confidence_threshold_high=0.65, 
                      confidence_threshold_medium=0.50):
    """
    Determine which signs to mark based on probabilities
    
    Args:
        probabilities: Dict with 'H', 'D', 'A' probabilities
        confidence_threshold_high: Threshold for single sign
        confidence_threshold_medium: Threshold for double sign
    
    Returns:
        String like '1', 'X', '1X', '12', etc.
    """
    home_prob = probabilities['H']
    draw_prob = probabilities['D']
    away_prob = probabilities['A']
    
    # Sort outcomes by probability
    outcomes = [
        ('H', home_prob, '1'),
        ('D', draw_prob, 'X'),
        ('A', away_prob, '2')
    ]
    outcomes.sort(key=lambda x: x[1], reverse=True)
    
    highest_outcome, highest_prob, highest_sign = outcomes[0]
    second_outcome, second_prob, second_sign = outcomes[1]
    
    # Strategy 1: Very confident single sign
    if highest_prob >= confidence_threshold_high:
        return highest_sign, "high_confidence", highest_prob
    
    # Strategy 2: Medium confidence - double sign (exclude least likely)
    elif highest_prob >= confidence_threshold_medium:
        # Mark two most likely outcomes
        signs = sorted([highest_sign, second_sign])
        return ''.join(signs), "medium_confidence", highest_prob + second_prob
    
    # Strategy 3: Low confidence - full coverage
    else:
        return '1X2', "low_confidence", 1.0


def create_stryktipset_coupon(predictions, strategy='balanced', max_cost=None):
    """
    Create a Stryktipset coupon with signing strategy
    
    Args:
        predictions: List of prediction dictionaries
        strategy: 'aggressive', 'balanced', or 'safe'
        max_cost: Maximum cost in SEK (e.g., 192 for 192 combinations)
    
    Returns:
        List of match signs with strategy
    """
    # Strategy thresholds
    thresholds = {
        'aggressive': {'high': 0.55, 'medium': 0.45},  # More single signs
        'balanced': {'high': 0.65, 'medium': 0.50},    # Balanced approach
        'safe': {'high': 0.75, 'medium': 0.60}         # More double/triple signs
    }
    
    high_thresh = thresholds[strategy]['high']
    medium_thresh = thresholds[strategy]['medium']
    
    coupon = []
    
    for i, pred in enumerate(predictions, 1):
        if not pred.get('final_prediction'):
            coupon.append({
                'match_num': i,
                'match': f"{pred['home_team']} vs {pred['away_team']}",
                'signs': '1X2',
                'strategy': 'no_data',
                'confidence': 0
            })
            continue
        
        # NEW: Try multiple sources for probabilities
        # First try direct 'probabilities' key (new league-specific predictor)
        probs = pred.get('probabilities')
        
        # Fallback to old format if not found
        if not probs:
            probs = pred.get('ensemble', {}).get('probabilities', 
                                                 pred.get('ml', {}).get('probabilities', {}))
        
        # If still no probabilities, create default ones
        if not probs or not isinstance(probs, dict):
            probs = {
                'H': 0.33,
                'D': 0.34,
                'A': 0.33,
            }
        
        # Convert numpy floats to regular floats if needed
        probs = {
            'H': float(probs.get('H', 0.33)),
            'D': float(probs.get('D', 0.34)),
            'A': float(probs.get('A', 0.33))
        }
        
        sign, strat, prob = get_sign_strategy(probs, high_thresh, medium_thresh)
        
        coupon.append({
            'match_num': i,
            'match': f"{pred['home_team']} vs {pred['away_team']}",
            'signs': sign,
            'strategy': strat,
            'confidence': prob,
            'probabilities': probs
        })
    
    # NEW: If max_cost specified, optimize the coupon to fit budget
    if max_cost:
        coupon = optimize_coupon_for_budget(coupon, max_cost)
    
    return coupon


def optimize_coupon_for_budget(coupon, max_cost):
    """
    Optimize coupon to fit within budget by reducing coverage on lowest confidence matches
    
    Args:
        coupon: Initial coupon
        max_cost: Maximum cost in SEK
    
    Returns:
        Optimized coupon within budget
    """
    import math
    
    # Calculate current cost
    stats = calculate_expected_combinations(coupon)
    current_cost = stats['cost_sek']
    
    if current_cost <= max_cost:
        print(f"✓ Coupon already within budget: {current_cost} SEK")
        return coupon
    
    print(f"\n🔧 OPTIMIZING COUPON FOR BUDGET")
    print(f"Current cost: {current_cost:,} SEK")
    print(f"Target cost: {max_cost:,} SEK")
    print(f"Need to reduce by: {current_cost - max_cost:,} SEK")
    
    # Sort matches by confidence (lowest first - these we'll reduce coverage on)
    sorted_indices = sorted(range(len(coupon)), 
                          key=lambda i: coupon[i].get('confidence', 0))
    
    optimized_coupon = [dict(match) for match in coupon]  # Deep copy
    
    # Try to reduce coverage starting with lowest confidence matches
    for idx in sorted_indices:
        stats = calculate_expected_combinations(optimized_coupon)
        if stats['cost_sek'] <= max_cost:
            break
        
        match = optimized_coupon[idx]
        current_signs = match['signs']
        
        # Reduction strategy: 1X2 → best double → best single
        if len(current_signs) == 3:  # 1X2
            # Reduce to best double sign
            probs = match.get('probabilities', {})
            outcomes = [
                ('H', probs.get('H', 0.33), '1'),
                ('D', probs.get('D', 0.34), 'X'),
                ('A', probs.get('A', 0.33), '2')
            ]
            outcomes.sort(key=lambda x: x[1], reverse=True)
            
            # Take two most likely
            sign1 = outcomes[0][2]
            sign2 = outcomes[1][2]
            new_signs = ''.join(sorted([sign1, sign2]))
            
            optimized_coupon[idx]['signs'] = new_signs
            optimized_coupon[idx]['strategy'] = 'budget_optimized'
            print(f"  Match {idx+1}: {current_signs} → {new_signs}")
            
        elif len(current_signs) == 2:  # Double sign
            # Reduce to best single sign
            probs = match.get('probabilities', {})
            outcomes = [
                ('H', probs.get('H', 0.33), '1'),
                ('D', probs.get('D', 0.34), 'X'),
                ('A', probs.get('A', 0.33), '2')
            ]
            outcomes.sort(key=lambda x: x[1], reverse=True)
            
            new_signs = outcomes[0][2]
            optimized_coupon[idx]['signs'] = new_signs
            optimized_coupon[idx]['strategy'] = 'budget_optimized'
            print(f"  Match {idx+1}: {current_signs} → {new_signs}")
    
    final_stats = calculate_expected_combinations(optimized_coupon)
    print(f"\n✓ Optimized cost: {final_stats['cost_sek']:,} SEK")
    print(f"  Single signs: {final_stats['single_signs']}")
    print(f"  Double signs: {final_stats['double_signs']}")
    print(f"  Triple signs: {final_stats['triple_signs']}")
    
    if final_stats['cost_sek'] > max_cost:
        print(f"\n⚠️  Warning: Could not reduce to {max_cost} SEK")
        print(f"   Minimum achievable: {final_stats['cost_sek']} SEK")
        print(f"   (All matches need at least one sign)")
    
    return optimized_coupon


def display_coupon(coupon, show_reasoning=True):
    """
    Display the Stryktipset coupon
    """
    print("\n" + "="*80)
    print("STRYKTIPSET COUPON")
    print("="*80 + "\n")
    
    single_signs = 0
    double_signs = 0
    triple_signs = 0
    
    for entry in coupon:
        signs = entry['signs']
        match = entry['match']
        
        print(f"{entry['match_num']:2d}. {match:40s} [{signs:3s}]", end="")
        
        if show_reasoning:
            conf = entry['confidence']
            strat = entry['strategy']
            print(f"  ({conf*100:3.0f}% - {strat})")
            
            if 'probabilities' in entry:
                probs = entry['probabilities']
                print(f"     1:{probs.get('H', 0)*100:3.0f}% | "
                      f"X:{probs.get('D', 0)*100:3.0f}% | "
                      f"2:{probs.get('A', 0)*100:3.0f}%")
        else:
            print()
        
        # Count sign types
        if len(signs) == 1:
            single_signs += 1
        elif len(signs) == 2:
            double_signs += 1
        else:
            triple_signs += 1
    
    print("\n" + "="*80)
    print(f"Single signs: {single_signs}")
    print(f"Double signs: {double_signs}")
    print(f"Triple signs: {triple_signs}")
    print(f"Total combinations: {2**single_signs * 2**double_signs * 3**triple_signs}")
    print("="*80 + "\n")


def calculate_expected_combinations(coupon):
    """
    Calculate number of combinations in the coupon
    """
    single = sum(1 for e in coupon if len(e['signs']) == 1)
    double = sum(1 for e in coupon if len(e['signs']) == 2)
    triple = sum(1 for e in coupon if len(e['signs']) == 3)
    
    combinations = (2**single) * (2**double) * (3**triple)
    
    return {
        'single_signs': single,
        'double_signs': double,
        'triple_signs': triple,
        'total_combinations': combinations,
        'cost_sek': combinations * 1  # 1 SEK per combination
    }


# Example usage
if __name__ == "__main__":
    # Example predictions (you'd get these from your predictor)
    example_predictions = [
        {
            'home_team': 'Chelsea',
            'away_team': 'Liverpool',
            'final_prediction': 'D',
            'ensemble': {
                'probabilities': {'H': 0.35, 'D': 0.40, 'A': 0.25}
            }
        },
        {
            'home_team': 'Arsenal',
            'away_team': 'West Ham',
            'final_prediction': 'H',
            'ensemble': {
                'probabilities': {'H': 0.70, 'D': 0.18, 'A': 0.12}
            }
        },
        {
            'home_team': 'Man United',
            'away_team': 'Man City',
            'final_prediction': 'A',
            'ensemble': {
                'probabilities': {'H': 0.20, 'D': 0.25, 'A': 0.55}
            }
        },
    ]
    
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    for strategy in ['aggressive', 'balanced', 'safe']:
        print(f"\n--- {strategy.upper()} STRATEGY ---")
        coupon = create_stryktipset_coupon(example_predictions, strategy=strategy)
        display_coupon(coupon, show_reasoning=True)
        
        stats = calculate_expected_combinations(coupon)
        print(f"Cost: {stats['cost_sek']} SEK for {stats['total_combinations']} combinations")