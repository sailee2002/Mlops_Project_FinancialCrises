"""
Enhanced Bias Detection Script
Checks model performance across critical economic conditions
Ensures fair scenario generation across different market regimes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ks_2samp
import json

# ============================================
# 1. DEFINE DATA SLICES
# ============================================

def create_data_slices(data):
    """
    Create comprehensive slices based on critical financial stress indicators
    Covers 7 dimensions: GDP, VIX, Unemployment, Yield Curve, Financial Stress, 
    Fed Rates, and Credit Stress
    """
    
    print("="*70)
    print("ENHANCED BIAS DETECTION - ECONOMIC REGIME SLICING")
    print("="*70 + "\n")
    
    slices = {}
    
    # ============================================
    # DIMENSION 1: GDP (Economic Size)
    # ============================================
    if 'GDP' in data.columns:
        gdp_25 = data['GDP'].quantile(0.25)
        gdp_75 = data['GDP'].quantile(0.75)
        
        slices['GDP_Low'] = data[data['GDP'] < gdp_25]
        slices['GDP_Medium'] = data[(data['GDP'] >= gdp_25) & (data['GDP'] < gdp_75)]
        slices['GDP_High'] = data[data['GDP'] >= gdp_75]
    
    # ============================================
    # DIMENSION 2: VIX (Market Volatility)
    # ============================================
    if 'VIX' in data.columns:
        vix_25 = data['VIX'].quantile(0.25)
        vix_75 = data['VIX'].quantile(0.75)
        
        slices['VIX_Low'] = data[data['VIX'] < vix_25]
        slices['VIX_Medium'] = data[(data['VIX'] >= vix_25) & (data['VIX'] < vix_75)]
        slices['VIX_High'] = data[data['VIX'] >= vix_75]
    
    # ============================================
    # DIMENSION 3: Unemployment (Labor Market)
    # ============================================
    if 'Unemployment_Rate' in data.columns:
        unemp_25 = data['Unemployment_Rate'].quantile(0.25)
        unemp_75 = data['Unemployment_Rate'].quantile(0.75)
        
        slices['Unemployment_Low'] = data[data['Unemployment_Rate'] < unemp_25]
        slices['Unemployment_Medium'] = data[(data['Unemployment_Rate'] >= unemp_25) & 
                                              (data['Unemployment_Rate'] < unemp_75)]
        slices['Unemployment_High'] = data[data['Unemployment_Rate'] >= unemp_75]
    
    # ============================================
    # DIMENSION 4: Yield Curve (Recession Signal) ⭐ NEW
    # ============================================
    if 'Yield_Curve_Spread' in data.columns:
        slices['Yield_Curve_Inverted'] = data[data['Yield_Curve_Spread'] < 0]
        slices['Yield_Curve_Normal'] = data[data['Yield_Curve_Spread'] >= 0]
        
        print("💡 Yield Curve Slicing:")
        print(f"   Inverted (Recession Signal): {len(slices['Yield_Curve_Inverted']):,} samples")
        print(f"   Normal: {len(slices['Yield_Curve_Normal']):,} samples\n")
    
    # ============================================
    # DIMENSION 5: Financial Stress Index ⭐ NEW
    # ============================================
    if 'Financial_Stress_Index' in data.columns:
        fsi_75 = data['Financial_Stress_Index'].quantile(0.75)
        
        slices['Financial_Stress_Low'] = data[data['Financial_Stress_Index'] < fsi_75]
        slices['Financial_Stress_High'] = data[data['Financial_Stress_Index'] >= fsi_75]
        
        print("💡 Financial Stress Slicing:")
        print(f"   Low Stress: {len(slices['Financial_Stress_Low']):,} samples")
        print(f"   High Stress: {len(slices['Financial_Stress_High']):,} samples\n")
    
    # ============================================
    # DIMENSION 6: Interest Rate Regime ⭐ NEW
    # ============================================
    if 'Federal_Funds_Rate' in data.columns:
        slices['Fed_Rate_ZeroLowerBound'] = data[data['Federal_Funds_Rate'] < 0.5]
        slices['Fed_Rate_Normal'] = data[(data['Federal_Funds_Rate'] >= 0.5) & 
                                          (data['Federal_Funds_Rate'] < 3.0)]
        slices['Fed_Rate_High'] = data[data['Federal_Funds_Rate'] >= 3.0]
        
        print("💡 Federal Funds Rate Regime:")
        print(f"   Zero Lower Bound (QE Era): {len(slices['Fed_Rate_ZeroLowerBound']):,} samples")
        print(f"   Normal (0.5-3%): {len(slices['Fed_Rate_Normal']):,} samples")
        print(f"   High (>3%): {len(slices['Fed_Rate_High']):,} samples\n")
    
    # ============================================
    # DIMENSION 7: Credit Stress ⭐ NEW
    # ============================================
    if 'High_Yield_Spread' in data.columns:
        hy_75 = data['High_Yield_Spread'].quantile(0.75)
        
        slices['Credit_Stress_Low'] = data[data['High_Yield_Spread'] < hy_75]
        slices['Credit_Stress_High'] = data[data['High_Yield_Spread'] >= hy_75]
        
        print("💡 Credit Market Stress:")
        print(f"   Low Stress: {len(slices['Credit_Stress_Low']):,} samples")
        print(f"   High Stress: {len(slices['Credit_Stress_High']):,} samples\n")
    
    # ============================================
    # SUMMARY
    # ============================================
    print("="*70)
    print("DATA SLICES CREATED")
    print("="*70)
    print(f"\n{'Slice Name':<35} {'Sample Count':>15}")
    print("-" * 70)
    
    for slice_name, slice_data in sorted(slices.items()):
        print(f"{slice_name:<35} {len(slice_data):>15,}")
    
    print(f"\n{'Total Slices:':<35} {len(slices):>15}")
    print("="*70 + "\n")
    
    return slices


# ============================================
# 2. EVALUATE SLICES
# ============================================

def evaluate_slice_performance(original_slices, generated_data, feature_list):
    """Evaluate model performance on each economic regime slice"""
    
    print("="*70)
    print("SLICE PERFORMANCE EVALUATION")
    print("="*70 + "\n")
    
    # Remove non-numeric columns from generated data
    gen_cols_to_drop = ['Scenario', 'Severity', 'Stress_Level', 'Stress_Score', 
                        'Crisis_Type', 'Crisis_Score']
    generated_numeric = generated_data.drop(columns=gen_cols_to_drop, errors='ignore')
    
    slice_results = {}
    
    for slice_name, original_slice in original_slices.items():
        print(f"📊 Evaluating: {slice_name}")
        
        # Get common features
        common_features = list(set(original_slice.columns) & 
                              set(generated_numeric.columns) & 
                              set(feature_list))
        
        if len(common_features) == 0:
            print(f"   ⚠️  No common features found\n")
            continue
        
        # Match generated scenarios to this slice
        matching_gen = match_generated_to_slice(slice_name, original_slice, 
                                                generated_numeric)
        
        if len(matching_gen) == 0:
            print(f"   ⚠️  No matching generated scenarios found\n")
            continue
        
        # Compute KS test for this slice
        ks_passed = 0
        ks_total = 0
        ks_pvalues = []
        
        for feature in common_features:
            try:
                _, pval = ks_2samp(original_slice[feature], matching_gen[feature])
                ks_total += 1
                ks_pvalues.append(pval)
                if pval > 0.05:
                    ks_passed += 1
            except Exception:
                continue
        
        ks_pass_rate = (ks_passed / ks_total * 100) if ks_total > 0 else 0
        avg_pvalue = np.mean(ks_pvalues) if ks_pvalues else 0
        
        slice_results[slice_name] = {
            'original_count': len(original_slice),
            'generated_count': len(matching_gen),
            'ks_pass_rate': ks_pass_rate,
            'ks_passed': ks_passed,
            'ks_total': ks_total,
            'avg_pvalue': avg_pvalue
        }
        
        # Status indicator
        if ks_pass_rate >= 75:
            status = "✅ EXCELLENT"
        elif ks_pass_rate >= 70:
            status = "✓ ACCEPTABLE"
        elif ks_pass_rate >= 60:
            status = "⚠️  MARGINAL"
        else:
            status = "❌ POOR"
        
        print(f"   Original: {len(original_slice):>6,} | Generated: {len(matching_gen):>6,}")
        print(f"   KS Pass Rate: {ks_pass_rate:>5.1f}% ({ks_passed}/{ks_total}) {status}\n")
    
    return slice_results


def match_generated_to_slice(slice_name, original_slice, generated_numeric):
    """Match generated scenarios to a specific slice based on its defining criteria"""
    
    # GDP-based slicing
    if 'GDP' in slice_name and 'GDP' in generated_numeric.columns and 'GDP' in original_slice.columns:
        if 'Low' in slice_name:
            threshold = original_slice['GDP'].max()
            return generated_numeric[generated_numeric['GDP'] < threshold]
        elif 'High' in slice_name:
            threshold = original_slice['GDP'].min()
            return generated_numeric[generated_numeric['GDP'] >= threshold]
        elif 'Medium' in slice_name:
            low_threshold = original_slice['GDP'].min()
            high_threshold = original_slice['GDP'].max()
            return generated_numeric[(generated_numeric['GDP'] >= low_threshold) & 
                                   (generated_numeric['GDP'] < high_threshold)]
    
    # VIX-based slicing
    elif 'VIX' in slice_name and 'VIX' in generated_numeric.columns and 'VIX' in original_slice.columns:
        if 'Low' in slice_name:
            threshold = original_slice['VIX'].max()
            return generated_numeric[generated_numeric['VIX'] < threshold]
        elif 'High' in slice_name:
            threshold = original_slice['VIX'].min()
            return generated_numeric[generated_numeric['VIX'] >= threshold]
        elif 'Medium' in slice_name:
            low_threshold = original_slice['VIX'].min()
            high_threshold = original_slice['VIX'].max()
            return generated_numeric[(generated_numeric['VIX'] >= low_threshold) & 
                                   (generated_numeric['VIX'] < high_threshold)]
    
    # Unemployment-based slicing
    elif 'Unemployment' in slice_name and 'Unemployment_Rate' in generated_numeric.columns:
        if 'Low' in slice_name:
            threshold = original_slice['Unemployment_Rate'].max()
            return generated_numeric[generated_numeric['Unemployment_Rate'] < threshold]
        elif 'High' in slice_name:
            threshold = original_slice['Unemployment_Rate'].min()
            return generated_numeric[generated_numeric['Unemployment_Rate'] >= threshold]
        elif 'Medium' in slice_name:
            low_threshold = original_slice['Unemployment_Rate'].min()
            high_threshold = original_slice['Unemployment_Rate'].max()
            return generated_numeric[(generated_numeric['Unemployment_Rate'] >= low_threshold) & 
                                   (generated_numeric['Unemployment_Rate'] < high_threshold)]
    
    # Yield Curve slicing ⭐ NEW
    elif 'Yield_Curve' in slice_name and 'Yield_Curve_Spread' in generated_numeric.columns:
        if 'Inverted' in slice_name:
            return generated_numeric[generated_numeric['Yield_Curve_Spread'] < 0]
        elif 'Normal' in slice_name:
            return generated_numeric[generated_numeric['Yield_Curve_Spread'] >= 0]
    
    # Financial Stress slicing ⭐ NEW
    elif 'Financial_Stress' in slice_name and 'Financial_Stress_Index' in generated_numeric.columns:
        threshold = original_slice['Financial_Stress_Index'].max() if 'Low' in slice_name else original_slice['Financial_Stress_Index'].min()
        if 'Low' in slice_name:
            return generated_numeric[generated_numeric['Financial_Stress_Index'] < threshold]
        else:
            return generated_numeric[generated_numeric['Financial_Stress_Index'] >= threshold]
    
    # Fed Rate regime slicing ⭐ NEW
    elif 'Fed_Rate' in slice_name and 'Federal_Funds_Rate' in generated_numeric.columns:
        if 'ZeroLowerBound' in slice_name:
            return generated_numeric[generated_numeric['Federal_Funds_Rate'] < 0.5]
        elif 'High' in slice_name:
            return generated_numeric[generated_numeric['Federal_Funds_Rate'] >= 3.0]
        elif 'Normal' in slice_name:
            return generated_numeric[(generated_numeric['Federal_Funds_Rate'] >= 0.5) & 
                                   (generated_numeric['Federal_Funds_Rate'] < 3.0)]
    
    # Credit Stress slicing ⭐ NEW
    elif 'Credit_Stress' in slice_name and 'High_Yield_Spread' in generated_numeric.columns:
        threshold = original_slice['High_Yield_Spread'].max() if 'Low' in slice_name else original_slice['High_Yield_Spread'].min()
        if 'Low' in slice_name:
            return generated_numeric[generated_numeric['High_Yield_Spread'] < threshold]
        else:
            return generated_numeric[generated_numeric['High_Yield_Spread'] >= threshold]
    
    # Default: return all generated data
    return generated_numeric


# ============================================
# 3. DETECT BIAS
# ============================================

def detect_bias(slice_results):
    """Detect performance disparities across economic regimes"""
    
    print("="*70)
    print("BIAS DETECTION ANALYSIS")
    print("="*70 + "\n")
    
    ks_pass_rates = [r['ks_pass_rate'] for r in slice_results.values()]
    
    if len(ks_pass_rates) == 0:
        print("⚠️  No slice results to analyze\n")
        return None
    
    mean_rate = np.mean(ks_pass_rates)
    std_rate = np.std(ks_pass_rates)
    min_rate = np.min(ks_pass_rates)
    max_rate = np.max(ks_pass_rates)
    range_rate = max_rate - min_rate
    
    # Find worst and best performing slices
    worst_slice = min(slice_results.items(), key=lambda x: x[1]['ks_pass_rate'])
    best_slice = max(slice_results.items(), key=lambda x: x[1]['ks_pass_rate'])
    
    print(f"Performance Summary:")
    print("-" * 70)
    print(f"  Mean KS Pass Rate:        {mean_rate:>6.1f}%")
    print(f"  Std Deviation:            {std_rate:>6.1f}%")
    print(f"  Min Pass Rate:            {min_rate:>6.1f}% ({worst_slice[0]})")
    print(f"  Max Pass Rate:            {max_rate:>6.1f}% ({best_slice[0]})")
    print(f"  Performance Range:        {range_rate:>6.1f}%\n")
    
    # Enhanced bias detection with multiple thresholds
    bias_detected = False
    bias_severity = "None"
    bias_level = 0  # 0-3 scale
    
    # Check if any slice is below absolute minimum threshold
    critical_failure = min_rate < 60
    
    if critical_failure:
        bias_detected = True
        bias_severity = "Critical"
        bias_level = 3
        print("🚨 CRITICAL BIAS DETECTED")
        print(f"   {worst_slice[0]} performance below 60% threshold")
        print(f"   This model is NOT suitable for stress testing")
    elif range_rate > 20:
        bias_detected = True
        bias_severity = "High"
        bias_level = 2
        print("⚠️  HIGH BIAS DETECTED")
        print(f"   Performance varies >{range_rate:.1f}% across economic regimes")
        print(f"   Model shows inconsistent quality across conditions")
    elif range_rate > 10:
        bias_detected = True
        bias_severity = "Moderate"
        bias_level = 1
        print("⚠️  MODERATE BIAS DETECTED")
        print(f"   Performance varies {range_rate:.1f}% across economic regimes")
        print(f"   Consider monitoring in production")
    else:
        print("✅ NO SIGNIFICANT BIAS DETECTED")
        print(f"   Performance is consistent across all economic regimes")
        print(f"   Model demonstrates robust scenario generation")
    
    # Additional context
    print(f"\nInterpretation:")
    print(f"  All slices above 70%: {'✅ Yes' if min_rate >= 70 else '❌ No'}")
    print(f"  Acceptable for production: {'✅ Yes' if min_rate >= 70 and range_rate < 25 else '❌ No'}")
    
    print("\n" + "="*70 + "\n")
    
    return {
        'bias_detected': bias_detected,
        'bias_severity': bias_severity,
        'bias_level': bias_level,
        'mean_rate': mean_rate,
        'std_rate': std_rate,
        'range_rate': range_rate,
        'min_rate': min_rate,
        'max_rate': max_rate,
        'worst_slice': worst_slice[0],
        'best_slice': best_slice[0],
        'critical_failure': critical_failure
    }


# ============================================
# 4. VISUALIZE BIAS
# ============================================

def create_bias_plots(slice_results, model_name='Dense_VAE_Optimized', output_dir='outputs/bias_detection'):
    """Create comprehensive bias detection visualizations"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    slice_names = list(slice_results.keys())
    ks_rates = [slice_results[s]['ks_pass_rate'] for s in slice_names]
    
    # Plot 1: Performance across all slices
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color coding based on performance
    colors = []
    for rate in ks_rates:
        if rate >= 75:
            colors.append('#2ecc71')  # Green - Excellent
        elif rate >= 70:
            colors.append('#3498db')  # Blue - Acceptable
        elif rate >= 60:
            colors.append('#f39c12')  # Orange - Marginal
        else:
            colors.append('#e74c3c')  # Red - Poor
    
    bars = ax.barh(slice_names, ks_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Reference lines
    ax.axvline(x=75, color='green', linestyle='--', label='Target (75%)', linewidth=2, alpha=0.7)
    ax.axvline(x=70, color='blue', linestyle='--', label='Minimum (70%)', linewidth=2, alpha=0.7)
    ax.axvline(x=np.mean(ks_rates), color='purple', linestyle='-', 
               label=f'Mean ({np.mean(ks_rates):.1f}%)', linewidth=2.5)
    
    # Add value labels
    for bar, rate in zip(bars, ks_rates):
        width = bar.get_width()
        label_x_pos = width + 1
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                f'{rate:.1f}%', va='center', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('KS Pass Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title(f'Economic Regime Performance Analysis\nModel: {model_name}', 
                 fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(100, max(ks_rates) + 5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bias_detection_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_dir}/bias_detection_{model_name}.png")
    
    # Plot 2: Performance distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(ks_rates, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(ks_rates), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(ks_rates):.1f}%')
    ax.axvline(np.median(ks_rates), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(ks_rates):.1f}%')
    
    ax.set_xlabel('KS Pass Rate (%)', fontweight='bold')
    ax.set_ylabel('Number of Slices', fontweight='bold')
    ax.set_title(f'Performance Distribution Across Economic Regimes\nModel: {model_name}', 
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_distribution_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_dir}/performance_distribution_{model_name}.png\n")


# ============================================
# 5. SAVE BIAS REPORT
# ============================================

def save_bias_report(model_name, slice_results, bias_analysis, output_dir='outputs/bias_detection'):
    """Save comprehensive bias detection report"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = f'{output_dir}/bias_report_{model_name}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ECONOMIC REGIME PERFORMANCE ANALYSIS\n")
        f.write("(Bias Detection for Financial Stress Testing)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Model: {model_name}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("="*70 + "\n")
        f.write("METHODOLOGY\n")
        f.write("="*70 + "\n\n")
        
        f.write("This analysis evaluates model performance across different economic\n")
        f.write("regimes to ensure consistent scenario generation quality during:\n\n")
        f.write("  1. GDP Levels (Economic Size)\n")
        f.write("  2. VIX Levels (Market Volatility)\n")
        f.write("  3. Unemployment Levels (Labor Market Stress)\n")
        f.write("  4. Yield Curve Status (Recession Indicator)\n")
        f.write("  5. Financial Stress Index (Systemic Risk)\n")
        f.write("  6. Federal Funds Rate (Monetary Policy Regime)\n")
        f.write("  7. Credit Stress (Corporate Bond Market)\n\n")
        
        f.write("Traditional demographic bias detection does not apply to macroeconomic\n")
        f.write("data. Instead, we measure performance equity across economic conditions.\n\n")
        
        f.write("="*70 + "\n")
        f.write("PERFORMANCE BY ECONOMIC REGIME\n")
        f.write("="*70 + "\n\n")
        
        # Sort by performance (worst to best)
        sorted_slices = sorted(slice_results.items(), key=lambda x: x[1]['ks_pass_rate'])
        
        for slice_name, results in sorted_slices:
            f.write(f"{slice_name}:\n")
            f.write(f"  Original Samples:     {results['original_count']:>6,}\n")
            f.write(f"  Generated Samples:    {results['generated_count']:>6,}\n")
            f.write(f"  KS Pass Rate:         {results['ks_pass_rate']:>6.1f}%\n")
            f.write(f"  Features Tested:      {results['ks_total']:>6}\n")
            
            if results['ks_pass_rate'] >= 75:
                f.write(f"  Status:               ✅ EXCELLENT\n\n")
            elif results['ks_pass_rate'] >= 70:
                f.write(f"  Status:               ✓ ACCEPTABLE\n\n")
            elif results['ks_pass_rate'] >= 60:
                f.write(f"  Status:               ⚠️  MARGINAL\n\n")
            else:
                f.write(f"  Status:               ❌ POOR\n\n")
        
        f.write("="*70 + "\n")
        f.write("BIAS ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        if bias_analysis:
            f.write(f"Bias Detected:          {bias_analysis['bias_detected']}\n")
            f.write(f"Severity Level:         {bias_analysis['bias_severity']}\n")
            f.write(f"Bias Score:             {bias_analysis['bias_level']}/3\n\n")
            
            f.write(f"Mean Performance:       {bias_analysis['mean_rate']:.1f}%\n")
            f.write(f"Performance Range:      {bias_analysis['range_rate']:.1f}%\n")
            f.write(f"Worst Regime:           {bias_analysis['worst_slice']} ({bias_analysis['min_rate']:.1f}%)\n")
            f.write(f"Best Regime:            {bias_analysis['best_slice']} ({bias_analysis['max_rate']:.1f}%)\n\n")
            
            # Recommendations
            f.write("="*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*70 + "\n\n")
            
            if bias_analysis['critical_failure']:
                f.write("🚨 CRITICAL ISSUES - MODEL NOT RECOMMENDED\n\n")
                f.write("This model shows critical performance degradation in specific\n")
                f.write("economic conditions. DO NOT deploy without addressing:\n\n")
                f.write("1. Retrain with additional data from weak regimes\n")
                f.write("2. Consider ensemble approach combining multiple models\n")
                f.write("3. Implement regime-specific post-processing corrections\n")
                
            elif bias_analysis['bias_severity'] == 'High':
                f.write("⚠️  HIGH BIAS - MITIGATION RECOMMENDED\n\n")
                f.write("1. Monitor performance closely in production\n")
                f.write("2. Consider retraining with balanced sampling\n")
                f.write("3. Implement slice-specific weight adjustments\n")
                f.write("4. Add ensemble methods for weak regimes\n")
                
            elif bias_analysis['bias_severity'] == 'Moderate':
                f.write("⚠️  MODERATE BIAS - ACCEPTABLE WITH MONITORING\n\n")
                f.write("1. Monitor slice performance in production\n")
                f.write("2. Document known weaknesses for risk managers\n")
                f.write("3. Consider minor data augmentation\n")
                
            else:
                f.write("✅ NO SIGNIFICANT BIAS DETECTED\n\n")
                f.write("Model demonstrates consistent performance across all economic\n")
                f.write("regimes. No mitigation required. Approved for deployment.\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("DEPLOYMENT RECOMMENDATION\n")
        f.write("="*70 + "\n\n")
        
        if bias_analysis:
            if bias_analysis['critical_failure']:
                f.write("[REJECT] Model requires significant improvement before deployment.\n")
            elif bias_analysis['min_rate'] >= 70 and bias_analysis['range_rate'] < 25:
                f.write("[APPROVE] Model meets minimum thresholds across all regimes.\n")
                f.write("Ready for production deployment with standard monitoring.\n")
            else:
                f.write("[CONDITIONAL] Model shows performance variance but meets minimum.\n")
                f.write("Deployment approved with enhanced monitoring on weak regimes.\n")
    
    print(f"✅ Saved bias report: {report_path}")
    
    # Save JSON for model selection
    json_path = f'{output_dir}/bias_analysis_{model_name}.json'
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(json_path, 'w') as f:
        json.dump({
            'model': model_name,
            'slice_results': convert_to_native(slice_results),
            'bias_analysis': convert_to_native(bias_analysis),
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2)
    
    print(f"✅ Saved bias analysis JSON: {json_path}\n")


# ============================================
# 6. MAIN EXECUTION
# ============================================

def main(model_name='Dense_VAE_Optimized'):
    """Main bias detection pipeline"""
    
    # Determine model directory and correct filenames
    if model_name == 'Dense_VAE_Optimized':
        model_dir = Path('outputs/output_Dense_VAE_optimized')
        scenarios_file = 'dense_vae_scenarios.csv'  # ✅ FIXED: Removed '_optimized'
    else:
        model_dir = Path('outputs/output_Ensemble_VAE')
        scenarios_file = 'ensemble_vae_scenarios.csv'
    
    # Load data
    print(f"Loading data for: {model_name}\n")
    scenarios = pd.read_csv(model_dir / scenarios_file)
    
    # Load original data - try multiple paths
    possible_paths = [
        'data/features/macro_features_clean.csv',
        '../../data/features/macro_features_clean.csv',
        '../../../data/features/macro_features_clean.csv',
        Path(__file__).parent.parent.parent / 'data/features/macro_features_clean.csv'
    ]
    
    original_data = None
    for data_path in possible_paths:
        try:
            original_data = pd.read_csv(data_path)
            print(f"✓ Found data at: {data_path}\n")
            break
        except FileNotFoundError:
            continue
    
    if original_data is None:
        raise FileNotFoundError(
            "Could not find macro_features_clean.csv\n" +
            "Please ensure the file exists in: data/features/macro_features_clean.csv"
        )
    
    if 'Date' in original_data.columns:
        original_data = original_data.drop('Date', axis=1)
    
    # ============================================
    # CRITICAL: Use hold-out test set only
    # ============================================
    
    print("⚠️  IMPORTANT: Using hold-out test set for bias detection!")
    print("Replicating training split...\n")
    
    from sklearn.model_selection import train_test_split
    
    # Use SAME split as training
    train_val, test_data = train_test_split(
        original_data,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Full data: {len(original_data):,} rows")
    print(f"Test set: {len(test_data):,} rows (hold-out) ✓\n")
    
    # Create slices from TEST DATA ONLY
    original_slices = create_data_slices(test_data)
    
    # Evaluate performance on slices
    feature_list = list(original_data.columns)
    slice_results = evaluate_slice_performance(original_slices, scenarios, feature_list)
    
    # Detect bias
    bias_analysis = detect_bias(slice_results)
    
    # Create visualizations
    create_bias_plots(slice_results)
    
    # Save report
    save_bias_report(model_name, slice_results, bias_analysis)
    
    print("="*60)
    print("✓ BIAS DETECTION COMPLETE")
    print("="*60)
    print(f"\nAnalyzed Model: {model_name}")
    print(f"Results saved in: outputs/bias_detection/")
    print("\nNext steps:")
    print("1. Review bias_detection_report.txt")
    print("2. Examine bias_detection_slices.png")
    print("3. Implement mitigation if needed")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    # Get model name from command line or use default
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'Dense_VAE_Optimized'
    
    main(model_name)