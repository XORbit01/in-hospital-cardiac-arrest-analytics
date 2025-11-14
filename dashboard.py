import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from io import BytesIO
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="IHCA Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('ihca_encoded.csv')
    # Handle missing values in cpr_duration_min (some are empty strings)
    df['cpr_duration_min'] = pd.to_numeric(df['cpr_duration_min'], errors='coerce')
    df['arrest_to_cpr_min'] = pd.to_numeric(df['arrest_to_cpr_min'], errors='coerce')
    # Remove outliers in CPR duration (likely data entry errors)
    df = df[df['cpr_duration_min'] <= 120].copy()  # Remove CPR > 120 minutes as outliers
    return df

df = load_data()

# Helper functions
def get_location_name(col):
    """Convert location column name to readable name"""
    return col.replace('loc_', '').replace('_', ' ').title()

def get_rhythm_name(col):
    """Convert rhythm column name to readable name"""
    if col == 'rhythm_Sinus/Other':
        return 'Sinus/Other'
    elif col == 'rhythm_nan':
        return 'Unknown'
    return col.replace('rhythm_', '').title()

# PDF Report Generation
def generate_pdf_report(df):
    """Generate a comprehensive PDF report of the IHCA analytics"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    # Container for the 'Flowable' objects
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    elements.append(Paragraph("IHCA Analytics Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                             styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Key Metrics
    total_cases = len(df)
    rosc_rate = df['rosc'].mean() * 100
    survival_24h_rate = df['survival_24h'].mean() * 100
    discharge_rate = df['survival_to_discharge'].mean() * 100
    median_age = df['age'].median()
    shockable_rate = df['shockable_rhythm'].mean() * 100
    icu_rate = df['loc_ICU'].mean() * 100
    
    # Gender distribution
    gender_valid = df[df['gender'].isin([0, 1])]
    male_pct = (gender_valid['gender'] == 1).mean() * 100 if len(gender_valid) > 0 else 0
    female_pct = (gender_valid['gender'] == 0).mean() * 100 if len(gender_valid) > 0 else 0
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Cases', f"{total_cases:,}"],
        ['Median Age', f"{median_age:.1f} years"],
        ['Gender Distribution', f"Male: {male_pct:.1f}%, Female: {female_pct:.1f}%"],
        ['ROSC Rate', f"{rosc_rate:.1f}%"],
        ['24h Survival Rate', f"{survival_24h_rate:.1f}%"],
        ['Discharge Survival Rate', f"{discharge_rate:.1f}%"],
        ['Shockable Rhythm Rate', f"{shockable_rate:.1f}%"],
        ['ICU Arrest Rate', f"{icu_rate:.1f}%"],
        ['Median CPR Duration', f"{df['cpr_duration_min'].median():.1f} minutes"],
        ['Median Arrest-to-CPR Delay', f"{df['arrest_to_cpr_min'].median():.1f} minutes"],
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Population Demographics
    elements.append(Paragraph("1. Population Demographics", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    age_stats = [
        ['Age Statistic', 'Value'],
        ['Mean Age', f"{df['age'].mean():.1f} years"],
        ['Median Age', f"{df['age'].median():.1f} years"],
        ['IQR (25th - 75th percentile)', 
         f"{df['age'].quantile(0.25):.1f} - {df['age'].quantile(0.75):.1f} years"],
        ['Skewness', f"{scipy_stats.skew(df['age'].dropna()):.2f}"],
    ]
    
    age_table = Table(age_stats, colWidths=[3*inch, 2.5*inch])
    age_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(age_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Comorbidities
    comorbidity_cols = [
        'coronary_artery_disease', 'heart_failure', 'heart_disease',
        'hypertension', 'copd', 'diabetes', 'cancer', 'covid_on_admission'
    ]
    comorbidity_names = [
        'CAD', 'Heart Failure', 'Heart Disease', 'Hypertension',
        'COPD', 'Diabetes', 'Cancer', 'COVID-19'
    ]
    
    comorb_data = [['Comorbidity', 'Prevalence (%)']]
    for col, name in zip(comorbidity_cols, comorbidity_names):
        prevalence = df[col].mean() * 100
        comorb_data.append([name, f"{prevalence:.1f}%"])
    
    comorb_table = Table(comorb_data, colWidths=[2.5*inch, 1.5*inch])
    comorb_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(comorb_table)
    elements.append(PageBreak())
    
    # Event Location
    elements.append(Paragraph("2. Event Location Analysis", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    location_cols = [col for col in df.columns if col.startswith('loc_')]
    location_data = [['Location', 'Cases', 'Percentage (%)', 'Survival Rate (%)']]
    
    for col in location_cols:
        loc_df = df[df[col] == 1]
        if len(loc_df) > 0:
            cases = len(loc_df)
            pct = (cases / len(df)) * 100
            survival = loc_df['survival_to_discharge'].mean() * 100
            location_data.append([
                get_location_name(col),
                str(cases),
                f"{pct:.1f}%",
                f"{survival:.1f}%"
            ])
    
    # Sort by cases
    location_data_sorted = [location_data[0]] + sorted(location_data[1:], 
                                                       key=lambda x: int(x[1]), 
                                                       reverse=True)
    
    loc_table = Table(location_data_sorted, colWidths=[2*inch, 1*inch, 1.2*inch, 1.2*inch])
    loc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(loc_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # ICU vs Non-ICU
    icu_df = df[df['loc_ICU'] == 1]
    non_icu_df = df[df['loc_ICU'] == 0]
    
    icu_comparison = [
        ['Location', 'Cases', 'ROSC Rate (%)', '24h Survival (%)', 'Discharge Survival (%)'],
        ['ICU', 
         len(icu_df),
         f"{icu_df['rosc'].mean()*100:.1f}%" if len(icu_df) > 0 else "N/A",
         f"{icu_df['survival_24h'].mean()*100:.1f}%" if len(icu_df) > 0 else "N/A",
         f"{icu_df['survival_to_discharge'].mean()*100:.1f}%" if len(icu_df) > 0 else "N/A"],
        ['Non-ICU',
         len(non_icu_df),
         f"{non_icu_df['rosc'].mean()*100:.1f}%",
         f"{non_icu_df['survival_24h'].mean()*100:.1f}%",
         f"{non_icu_df['survival_to_discharge'].mean()*100:.1f}%"]
    ]
    
    icu_table = Table(icu_comparison, colWidths=[1.2*inch, 1*inch, 1.2*inch, 1.2*inch, 1.5*inch])
    icu_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(icu_table)
    elements.append(PageBreak())
    
    # Initial Rhythm
    elements.append(Paragraph("3. Initial Rhythm Analysis", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    rhythm_cols = [col for col in df.columns if col.startswith('rhythm_')]
    rhythm_data = [['Rhythm', 'Cases', 'Percentage (%)', 'ROSC Rate (%)', 'Survival Rate (%)']]
    
    for col in rhythm_cols:
        rhythm_df = df[df[col] == 1]
        if len(rhythm_df) > 0:
            cases = len(rhythm_df)
            pct = (cases / len(df)) * 100
            rosc = rhythm_df['rosc'].mean() * 100
            survival = rhythm_df['survival_to_discharge'].mean() * 100
            rhythm_data.append([
                get_rhythm_name(col),
                str(cases),
                f"{pct:.1f}%",
                f"{rosc:.1f}%",
                f"{survival:.1f}%"
            ])
    
    rhythm_data_sorted = [rhythm_data[0]] + sorted(rhythm_data[1:], 
                                                    key=lambda x: int(x[1]), 
                                                    reverse=True)
    
    rhythm_table = Table(rhythm_data_sorted, colWidths=[1.5*inch, 0.8*inch, 1*inch, 1*inch, 1.2*inch])
    rhythm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(rhythm_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Shockable vs Non-shockable
    shockable_df = df[df['shockable_rhythm'] == 1]
    non_shockable_df = df[df['shockable_rhythm'] == 0]
    
    shockable_comparison = [
        ['Rhythm Type', 'Cases', 'ROSC Rate (%)', '24h Survival (%)', 'Discharge Survival (%)'],
        ['Shockable (VF/VT)',
         len(shockable_df),
         f"{shockable_df['rosc'].mean()*100:.1f}%" if len(shockable_df) > 0 else "N/A",
         f"{shockable_df['survival_24h'].mean()*100:.1f}%" if len(shockable_df) > 0 else "N/A",
         f"{shockable_df['survival_to_discharge'].mean()*100:.1f}%" if len(shockable_df) > 0 else "N/A"],
        ['Non-Shockable',
         len(non_shockable_df),
         f"{non_shockable_df['rosc'].mean()*100:.1f}%",
         f"{non_shockable_df['survival_24h'].mean()*100:.1f}%",
         f"{non_shockable_df['survival_to_discharge'].mean()*100:.1f}%"]
    ]
    
    shockable_table = Table(shockable_comparison, colWidths=[1.5*inch, 1*inch, 1.2*inch, 1.2*inch, 1.5*inch])
    shockable_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(shockable_table)
    elements.append(PageBreak())
    
    # Interventions
    elements.append(Paragraph("4. Interventions Analysis", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    interventions_data = [
        ['Metric', 'Value'],
        ['Patients Receiving Shocks', f"{(df['shock_count'] > 0).sum()} ({(df['shock_count'] > 0).mean()*100:.1f}%)"],
        ['Mean Shock Count', f"{df[df['shock_count'] > 0]['shock_count'].mean():.1f}" if (df['shock_count'] > 0).sum() > 0 else "N/A"],
        ['Max Shock Count', f"{df['shock_count'].max()}"],
        ['Mean Max Energy', f"{df[df['max_energy'] > 0]['max_energy'].mean():.0f} J" if (df['max_energy'] > 0).sum() > 0 else "N/A"],
        ['Median CPR Duration', f"{df['cpr_duration_min'].median():.1f} minutes"],
        ['Mean CPR Duration', f"{df['cpr_duration_min'].mean():.1f} minutes"],
        ['IQR CPR Duration', f"{df['cpr_duration_min'].quantile(0.25):.1f} - {df['cpr_duration_min'].quantile(0.75):.1f} minutes"],
        ['Median Arrest-to-CPR Delay', f"{df['arrest_to_cpr_min'].median():.1f} minutes"],
        ['Mean Arrest-to-CPR Delay', f"{df['arrest_to_cpr_min'].mean():.1f} minutes"],
        ['Delayed CPR (>2 min)', f"{(df['arrest_to_cpr_min'] > 2).sum()} ({(df['arrest_to_cpr_min'] > 2).mean()*100:.1f}%)"],
    ]
    
    interventions_table = Table(interventions_data, colWidths=[3*inch, 2.5*inch])
    interventions_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(interventions_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # CPR Duration by Outcome
    survived_cpr = df[df['survival_to_discharge'] == 1]['cpr_duration_min']
    died_cpr = df[df['survival_to_discharge'] == 0]['cpr_duration_min']
    
    cpr_outcome_data = [
        ['Outcome', 'Median CPR (min)', 'Mean CPR (min)', 'IQR (min)'],
        ['Survived',
         f"{survived_cpr.median():.1f}" if len(survived_cpr) > 0 else "N/A",
         f"{survived_cpr.mean():.1f}" if len(survived_cpr) > 0 else "N/A",
         f"{survived_cpr.quantile(0.25):.1f} - {survived_cpr.quantile(0.75):.1f}" if len(survived_cpr) > 0 else "N/A"],
        ['Died',
         f"{died_cpr.median():.1f}" if len(died_cpr) > 0 else "N/A",
         f"{died_cpr.mean():.1f}" if len(died_cpr) > 0 else "N/A",
         f"{died_cpr.quantile(0.25):.1f} - {died_cpr.quantile(0.75):.1f}" if len(died_cpr) > 0 else "N/A"]
    ]
    
    cpr_outcome_table = Table(cpr_outcome_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 2*inch])
    cpr_outcome_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(cpr_outcome_table)
    elements.append(PageBreak())
    
    # Outcomes Summary
    elements.append(Paragraph("5. Outcomes Summary", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    outcomes_summary = [
        ['Outcome Metric', 'Rate (%)', 'Cases'],
        ['ROSC', f"{rosc_rate:.1f}%", f"{df['rosc'].sum()}"],
        ['24h Survival', f"{survival_24h_rate:.1f}%", f"{df['survival_24h'].sum()}"],
        ['Discharge Survival', f"{discharge_rate:.1f}%", f"{df['survival_to_discharge'].sum()}"],
    ]
    
    outcomes_table = Table(outcomes_summary, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    outcomes_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(outcomes_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Key Insights
    elements.append(Paragraph("Key Insights", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    insights = [
        f"• Total of {total_cases:,} in-hospital cardiac arrest cases analyzed.",
        f"• Overall ROSC rate: {rosc_rate:.1f}%, with {survival_24h_rate:.1f}% surviving 24 hours and {discharge_rate:.1f}% surviving to discharge.",
        f"• Shockable rhythms (VF/VT) represent {shockable_rate:.1f}% of all cases.",
        f"• ICU arrests account for {icu_rate:.1f}% of all cardiac arrests.",
        f"• Median CPR duration: {df['cpr_duration_min'].median():.1f} minutes.",
        f"• Median arrest-to-CPR delay: {df['arrest_to_cpr_min'].median():.1f} minutes.",
        f"• {(df['arrest_to_cpr_min'] > 2).sum()} cases ({((df['arrest_to_cpr_min'] > 2).mean()*100):.1f}%) had delayed CPR (>2 minutes).",
    ]
    
    for insight in insights:
        elements.append(Paragraph(insight, styles['Normal']))
        elements.append(Spacer(1, 0.05*inch))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Sidebar navigation
st.sidebar.title("🏥 IHCA Analytics Dashboard")
st.sidebar.markdown("---")

# PDF Report Download Button
st.sidebar.markdown("### 📄 Generate Report")

# Initialize session state for PDF (if not already initialized)
if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None
if 'pdf_timestamp' not in st.session_state:
    st.session_state.pdf_timestamp = None

# Button to generate PDF
if st.sidebar.button("📥 Generate PDF Report", use_container_width=True, key="gen_pdf"):
    with st.sidebar:
        with st.spinner("Generating PDF report... This may take a few moments."):
            pdf_buffer = generate_pdf_report(df)
            st.session_state.pdf_bytes = pdf_buffer.getvalue()
            st.session_state.pdf_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.success("PDF report generated!")

# Download button (only shows if PDF is generated)
if st.session_state.get('pdf_bytes') is not None:
    with st.sidebar:
        st.download_button(
            label="⬇️ Download PDF Report",
            data=st.session_state.pdf_bytes,
            file_name=f"IHCA_Analytics_Report_{st.session_state.pdf_timestamp}.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Download the generated PDF report"
        )
        st.caption("Report includes: Executive Summary, Demographics, Location Analysis, Rhythm Analysis, Interventions, and Outcomes")
else:
    st.sidebar.info("👆 Click 'Generate PDF Report' to create a comprehensive PDF report.")

st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Select Dimension",
    [
        "📊 Overview",
        "👥 1. Population Demographics",
        "📍 2. Event Location",
        "💓 3. Initial Rhythm",
        "⚡ 4. Interventions",
        "✅ 5. Outcomes",
        "⏱️ 6. Temporal Patterns",
        "🔗 7. Correlation Analysis",
        "🔍 8. Rare Events"
    ]
)

# Overview Page
if page == "📊 Overview":
    st.title("🏥 IHCA Analytics Dashboard - Overview")
    st.markdown("### Comprehensive In-Hospital Cardiac Arrest Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cases", len(df))
    with col2:
        st.metric("ROSC Rate", f"{df['rosc'].mean()*100:.1f}%")
    with col3:
        st.metric("24h Survival", f"{df['survival_24h'].mean()*100:.1f}%")
    with col4:
        st.metric("Discharge Survival", f"{df['survival_to_discharge'].mean()*100:.1f}%")
    
    st.markdown("---")
    
    # Key statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Key Statistics")
        st.write(f"- **Median Age**: {df['age'].median():.1f} years")
        st.write(f"- **IQR Age**: {df['age'].quantile(0.25):.1f} - {df['age'].quantile(0.75):.1f} years")
        # Calculate gender distribution excluding unknown values (-1)
        gender_valid = df[df['gender'].isin([0, 1])]
        male_pct = (gender_valid['gender'] == 1).mean() * 100 if len(gender_valid) > 0 else 0
        female_pct = (gender_valid['gender'] == 0).mean() * 100 if len(gender_valid) > 0 else 0
        st.write(f"- **Gender Distribution**: {male_pct:.1f}% Male, {female_pct:.1f}% Female")
        st.write(f"- **Shockable Rhythm**: {df['shockable_rhythm'].mean()*100:.1f}%")
        st.write(f"- **Median CPR Duration**: {df['cpr_duration_min'].median():.1f} minutes")
        st.write(f"- **ICU Arrests**: {df['loc_ICU'].mean()*100:.1f}%")
    
    with col2:
        st.subheader("🎯 Quick Insights")
        shockable_survival = df[df['shockable_rhythm']==1]['survival_to_discharge'].mean() if df['shockable_rhythm'].sum() > 0 else 0
        non_shockable_survival = df[df['shockable_rhythm']==0]['survival_to_discharge'].mean()
        st.write(f"- **Shockable Rhythm Survival**: {shockable_survival*100:.1f}%")
        st.write(f"- **Non-Shockable Rhythm Survival**: {non_shockable_survival*100:.1f}%")
        
        icu_survival = df[df['loc_ICU']==1]['survival_to_discharge'].mean() if df['loc_ICU'].sum() > 0 else 0
        non_icu_survival = df[df['loc_ICU']==0]['survival_to_discharge'].mean()
        st.write(f"- **ICU Survival**: {icu_survival*100:.1f}%")
        st.write(f"- **Non-ICU Survival**: {non_icu_survival*100:.1f}%")
        
        delayed_cpr = (df['arrest_to_cpr_min'] > 2).sum()
        st.write(f"- **Delayed CPR (>2 min)**: {delayed_cpr} cases ({delayed_cpr/len(df)*100:.1f}%)")

# 1. Population Demographics
elif page == "👥 1. Population Demographics":
    st.title("👥 Population Demographics")
    st.markdown("### Goal: Who are the patients?")
    
    # Age Distribution
    st.subheader("📊 Age Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Histogram
        fig = px.histogram(
            df, 
            x='age', 
            nbins=30,
            title="Age Distribution",
            labels={'age': 'Age (years)', 'count': 'Frequency'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age Boxplot by Survival
        df_survival = df.copy()
        df_survival['Survival Status'] = df_survival['survival_to_discharge'].map({1: 'Survived', 0: 'Died'})
        fig = px.box(
            df_survival,
            x='Survival Status',
            y='age',
            title="Age Distribution by Survival Status",
            labels={'age': 'Age (years)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Age Statistics
    st.subheader("📈 Age Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Age", f"{df['age'].mean():.1f} years")
    with col2:
        st.metric("Median Age", f"{df['age'].median():.1f} years")
    with col3:
        q1, q3 = df['age'].quantile(0.25), df['age'].quantile(0.75)
        st.metric("IQR", f"{q1:.1f} - {q3:.1f} years")
    with col4:
        skewness = scipy_stats.skew(df['age'].dropna())
        st.metric("Skewness", f"{skewness:.2f}")
    
    # Age and Outcomes
    st.subheader("🔗 Age Relationship with Outcomes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Age vs ROSC
        df_age_rosc = df.groupby(pd.cut(df['age'], bins=10))['rosc'].mean().reset_index()
        df_age_rosc['age_mid'] = df_age_rosc['age'].apply(lambda x: x.mid)
        fig = px.bar(
            df_age_rosc,
            x='age_mid',
            y='rosc',
            title="ROSC Rate by Age Group",
            labels={'age_mid': 'Age (years)', 'rosc': 'ROSC Rate'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age vs 24h Survival
        df_age_24h = df.groupby(pd.cut(df['age'], bins=10))['survival_24h'].mean().reset_index()
        df_age_24h['age_mid'] = df_age_24h['age'].apply(lambda x: x.mid)
        fig = px.bar(
            df_age_24h,
            x='age_mid',
            y='survival_24h',
            title="24h Survival Rate by Age Group",
            labels={'age_mid': 'Age (years)', 'survival_24h': '24h Survival Rate'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Age vs Discharge Survival
        df_age_discharge = df.groupby(pd.cut(df['age'], bins=10))['survival_to_discharge'].mean().reset_index()
        df_age_discharge['age_mid'] = df_age_discharge['age'].apply(lambda x: x.mid)
        fig = px.bar(
            df_age_discharge,
            x='age_mid',
            y='survival_to_discharge',
            title="Discharge Survival Rate by Age Group",
            labels={'age_mid': 'Age (years)', 'survival_to_discharge': 'Discharge Survival Rate'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Gender Distribution
    st.subheader("⚧️ Gender Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender_counts = df['gender'].value_counts().sort_index()
        # Map gender values to labels, handling -1 (unknown)
        gender_map = {-1: 'Unknown', 0: 'Female', 1: 'Male'}
        gender_labels = [gender_map.get(val, f'Unknown ({val})') for val in gender_counts.index]
        fig = px.pie(
            values=gender_counts.values,
            names=gender_labels,
            title="Gender Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Survival by Gender
        gender_survival = df.groupby('gender').agg({
            'rosc': 'mean',
            'survival_24h': 'mean',
            'survival_to_discharge': 'mean'
        }).reset_index()
        gender_survival['gender'] = gender_survival['gender'].map({-1: 'Unknown', 0: 'Female', 1: 'Male'})
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='ROSC', x=gender_survival['gender'], y=gender_survival['rosc']*100))
        fig.add_trace(go.Bar(name='24h Survival', x=gender_survival['gender'], y=gender_survival['survival_24h']*100))
        fig.add_trace(go.Bar(name='Discharge Survival', x=gender_survival['gender'], y=gender_survival['survival_to_discharge']*100))
        fig.update_layout(
            title="Survival Rates by Gender",
            yaxis_title="Rate (%)",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Comorbidities
    st.subheader("🏥 Comorbidities Analysis")
    
    comorbidity_cols = [
        'coronary_artery_disease', 'heart_failure', 'heart_disease',
        'hypertension', 'copd', 'diabetes', 'cancer', 'covid_on_admission'
    ]
    comorbidity_names = [
        'CAD', 'Heart Failure', 'Heart Disease', 'Hypertension',
        'COPD', 'Diabetes', 'Cancer', 'COVID-19'
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Comorbidity prevalence
        comorbidity_prev = [df[col].mean() for col in comorbidity_cols]
        fig = px.bar(
            x=comorbidity_names,
            y=comorbidity_prev,
            title="Comorbidity Prevalence",
            labels={'x': 'Comorbidity', 'y': 'Prevalence'}
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Survival rates by comorbidity
        survival_by_comorb = []
        for col in comorbidity_cols:
            with_comorb = df[df[col]==1]['survival_to_discharge'].mean()
            without_comorb = df[df[col]==0]['survival_to_discharge'].mean()
            survival_by_comorb.append({
                'comorbidity': col,
                'with': with_comorb,
                'without': without_comorb
            })
        
        df_comorb_surv = pd.DataFrame(survival_by_comorb)
        df_comorb_surv['comorbidity'] = comorbidity_names
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='With Comorbidity', x=df_comorb_surv['comorbidity'], y=df_comorb_surv['with']*100))
        fig.add_trace(go.Bar(name='Without Comorbidity', x=df_comorb_surv['comorbidity'], y=df_comorb_surv['without']*100))
        fig.update_layout(
            title="Survival Rates by Comorbidity",
            yaxis_title="Discharge Survival Rate (%)",
            xaxis_tickangle=-45,
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# 2. Event Location
elif page == "📍 2. Event Location":
    st.title("📍 Event Location Analysis")
    st.markdown("### Goal: Where do cardiac arrests occur?")
    
    # Location columns
    location_cols = [col for col in df.columns if col.startswith('loc_')]
    location_names = [get_location_name(col) for col in location_cols]
    
    # Location distribution
    st.subheader("📊 Event Location Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        location_counts = [df[col].sum() for col in location_cols]
        fig = px.bar(
            x=location_names,
            y=location_counts,
            title="Number of Arrests by Location",
            labels={'x': 'Location', 'y': 'Number of Cases'}
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        location_percentages = [df[col].mean()*100 for col in location_cols]
        fig = px.pie(
            values=location_percentages,
            names=location_names,
            title="Percentage Distribution by Location"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # ICU vs Non-ICU
    st.subheader("🏥 ICU vs Non-ICU Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    icu_cases = df['loc_ICU'].sum()
    non_icu_cases = len(df) - icu_cases
    
    with col1:
        st.metric("ICU Cases", f"{icu_cases} ({icu_cases/len(df)*100:.1f}%)")
    with col2:
        st.metric("Non-ICU Cases", f"{non_icu_cases} ({non_icu_cases/len(df)*100:.1f}%)")
    with col3:
        icu_survival = df[df['loc_ICU']==1]['survival_to_discharge'].mean() if icu_cases > 0 else 0
        st.metric("ICU Survival", f"{icu_survival*100:.1f}%")
    with col4:
        non_icu_survival = df[df['loc_ICU']==0]['survival_to_discharge'].mean()
        st.metric("Non-ICU Survival", f"{non_icu_survival*100:.1f}%")
    
    # ICU vs Non-ICU Survival Comparison
    st.subheader("📈 Survival Comparison: ICU vs Non-ICU")
    
    icu_df = df[df['loc_ICU']==1]
    non_icu_df = df[df['loc_ICU']==0]
    
    if len(icu_df) > 0 and len(non_icu_df) > 0:
        comparison_data = {
            'Location': ['ICU', 'Non-ICU'],
            'ROSC': [
                icu_df['rosc'].mean(),
                non_icu_df['rosc'].mean()
            ],
            '24h Survival': [
                icu_df['survival_24h'].mean(),
                non_icu_df['survival_24h'].mean()
            ],
            'Discharge Survival': [
                icu_df['survival_to_discharge'].mean(),
                non_icu_df['survival_to_discharge'].mean()
            ]
        }
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='ROSC', x=comparison_data['Location'], y=[x*100 for x in comparison_data['ROSC']]))
        fig.add_trace(go.Bar(name='24h Survival', x=comparison_data['Location'], y=[x*100 for x in comparison_data['24h Survival']]))
        fig.add_trace(go.Bar(name='Discharge Survival', x=comparison_data['Location'], y=[x*100 for x in comparison_data['Discharge Survival']]))
        fig.update_layout(
            title="Outcome Rates: ICU vs Non-ICU",
            yaxis_title="Rate (%)",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Location × Outcomes Heatmap
    st.subheader("🔥 Location × Outcomes Heatmap")
    
    location_outcomes = []
    for col in location_cols:
        loc_df = df[df[col]==1]
        if len(loc_df) > 0:
            location_outcomes.append({
                'Location': get_location_name(col),
                'Cases': len(loc_df),
                'ROSC Rate': loc_df['rosc'].mean(),
                '24h Survival': loc_df['survival_24h'].mean(),
                'Discharge Survival': loc_df['survival_to_discharge'].mean()
            })
    
    df_location_outcomes = pd.DataFrame(location_outcomes)
    df_location_outcomes = df_location_outcomes.sort_values('Cases', ascending=False)
    
    # Heatmap
    heatmap_data = df_location_outcomes.set_index('Location')[['ROSC Rate', '24h Survival', 'Discharge Survival']]
    fig = px.imshow(
        heatmap_data.T,
        labels=dict(x="Location", y="Outcome", color="Rate"),
        x=heatmap_data.index,
        y=['ROSC Rate', '24h Survival', 'Discharge Survival'],
        color_continuous_scale='RdYlGn',
        aspect="auto",
        title="Survival Rates by Location"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Shockable rhythms by location
    st.subheader("⚡ Shockable Rhythms by Location")
    
    shockable_by_loc = []
    for col in location_cols:
        loc_df = df[df[col]==1]
        if len(loc_df) > 0:
            shockable_rate = loc_df['shockable_rhythm'].mean()
            shockable_by_loc.append({
                'Location': get_location_name(col),
                'Shockable Rate': shockable_rate,
                'Cases': len(loc_df)
            })
    
    df_shockable_loc = pd.DataFrame(shockable_by_loc)
    df_shockable_loc = df_shockable_loc.sort_values('Shockable Rate', ascending=False)
    
    fig = px.bar(
        df_shockable_loc,
        x='Location',
        y='Shockable Rate',
        title="Shockable Rhythm Rate by Location",
        labels={'Shockable Rate': 'Shockable Rhythm Rate'}
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# 3. Initial Rhythm
elif page == "💓 3. Initial Rhythm":
    st.title("💓 Initial Rhythm Analysis")
    st.markdown("### Goal: Rhythm type determines survival")
    
    # Rhythm columns
    rhythm_cols = [col for col in df.columns if col.startswith('rhythm_')]
    rhythm_names = [get_rhythm_name(col) for col in rhythm_cols]
    
    # Rhythm distribution
    st.subheader("📊 Initial Rhythm Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rhythm_counts = [df[col].sum() for col in rhythm_cols]
        fig = px.bar(
            x=rhythm_names,
            y=rhythm_counts,
            title="Number of Cases by Initial Rhythm",
            labels={'x': 'Rhythm', 'y': 'Number of Cases'}
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        rhythm_percentages = [df[col].mean()*100 for col in rhythm_cols]
        fig = px.pie(
            values=rhythm_percentages,
            names=rhythm_names,
            title="Percentage Distribution by Rhythm"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Shockable vs Non-shockable
    st.subheader("⚡ Shockable vs Non-Shockable Rhythms")
    
    col1, col2, col3, col4 = st.columns(4)
    
    shockable_cases = df['shockable_rhythm'].sum()
    non_shockable_cases = len(df) - shockable_cases
    
    with col1:
        st.metric("Shockable (VF/VT)", f"{shockable_cases} ({shockable_cases/len(df)*100:.1f}%)")
    with col2:
        st.metric("Non-Shockable", f"{non_shockable_cases} ({non_shockable_cases/len(df)*100:.1f}%)")
    with col3:
        shockable_survival = df[df['shockable_rhythm']==1]['survival_to_discharge'].mean() if shockable_cases > 0 else 0
        st.metric("Shockable Survival", f"{shockable_survival*100:.1f}%")
    with col4:
        non_shockable_survival = df[df['shockable_rhythm']==0]['survival_to_discharge'].mean()
        st.metric("Non-Shockable Survival", f"{non_shockable_survival*100:.1f}%")
    
    # ROSC Rate by Rhythm
    st.subheader("📈 ROSC Rate by Rhythm")
    
    rhythm_rosc = []
    for col in rhythm_cols:
        rhythm_df = df[df[col]==1]
        if len(rhythm_df) > 0:
            rhythm_rosc.append({
                'Rhythm': get_rhythm_name(col),
                'ROSC Rate': rhythm_df['rosc'].mean(),
                'Cases': len(rhythm_df)
            })
    
    df_rhythm_rosc = pd.DataFrame(rhythm_rosc)
    df_rhythm_rosc = df_rhythm_rosc.sort_values('ROSC Rate', ascending=False)
    
    fig = px.bar(
        df_rhythm_rosc,
        x='Rhythm',
        y='ROSC Rate',
        title="ROSC Rate by Initial Rhythm",
        labels={'ROSC Rate': 'ROSC Rate'}
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Survival by Rhythm
    st.subheader("✅ Survival Rates by Rhythm")
    
    rhythm_survival = []
    for col in rhythm_cols:
        rhythm_df = df[df[col]==1]
        if len(rhythm_df) > 0:
            rhythm_survival.append({
                'Rhythm': get_rhythm_name(col),
                '24h Survival': rhythm_df['survival_24h'].mean(),
                'Discharge Survival': rhythm_df['survival_to_discharge'].mean(),
                'Cases': len(rhythm_df)
            })
    
    df_rhythm_survival = pd.DataFrame(rhythm_survival)
    df_rhythm_survival = df_rhythm_survival.sort_values('Discharge Survival', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='24h Survival', x=df_rhythm_survival['Rhythm'], y=df_rhythm_survival['24h Survival']*100))
    fig.add_trace(go.Bar(name='Discharge Survival', x=df_rhythm_survival['Rhythm'], y=df_rhythm_survival['Discharge Survival']*100))
    fig.update_layout(
        title="Survival Rates by Initial Rhythm",
        yaxis_title="Survival Rate (%)",
        xaxis_tickangle=-45,
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Rhythm × Location Stacked Bar
    st.subheader("📍 Rhythm × Location Analysis")
    
    # Get top locations
    location_cols = [col for col in df.columns if col.startswith('loc_')]
    location_counts = {col: df[col].sum() for col in location_cols}
    top_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_location_cols = [loc[0] for loc in top_locations]
    
    # Create stacked bar chart
    rhythm_loc_data = []
    for rhythm_col in rhythm_cols:
        rhythm_name = get_rhythm_name(rhythm_col)
        for loc_col in top_location_cols:
            loc_name = get_location_name(loc_col)
            count = len(df[(df[rhythm_col]==1) & (df[loc_col]==1)])
            if count > 0:
                rhythm_loc_data.append({
                    'Rhythm': rhythm_name,
                    'Location': loc_name,
                    'Count': count
                })
    
    if rhythm_loc_data:
        df_rhythm_loc = pd.DataFrame(rhythm_loc_data)
        fig = px.bar(
            df_rhythm_loc,
            x='Location',
            y='Count',
            color='Rhythm',
            title="Rhythm Distribution by Location (Top 5 Locations)",
            barmode='stack'
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# 4. Interventions
elif page == "⚡ 4. Interventions":
    st.title("⚡ Interventions Analysis")
    st.markdown("### Goal: Did they receive shocks or CPR early enough?")
    
    # Shocks
    st.subheader("🔌 Shock Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    patients_shocked = (df['shock_count'] > 0).sum()
    with col1:
        st.metric("Patients Receiving Shocks", f"{patients_shocked} ({patients_shocked/len(df)*100:.1f}%)")
    with col2:
        st.metric("Mean Shock Count", f"{df[df['shock_count']>0]['shock_count'].mean():.1f}" if patients_shocked > 0 else "N/A")
    with col3:
        st.metric("Max Shock Count", f"{df['shock_count'].max()}")
    with col4:
        st.metric("Mean Max Energy", f"{df[df['max_energy']>0]['max_energy'].mean():.0f} J" if (df['max_energy']>0).sum() > 0 else "N/A")
    
    # Shock count distribution
    col1, col2 = st.columns(2)
    
    with col1:
        shock_dist = df[df['shock_count'] > 0]['shock_count'].value_counts().sort_index()
        if len(shock_dist) > 0:
            fig = px.bar(
                x=shock_dist.index,
                y=shock_dist.values,
                title="Shock Count Distribution",
                labels={'x': 'Number of Shocks', 'y': 'Number of Patients'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Shock count vs survival
        shock_survival = df.groupby(df['shock_count'].clip(upper=5))['survival_to_discharge'].mean().reset_index()
        shock_survival.columns = ['Shock Count', 'Survival Rate']
        fig = px.bar(
            shock_survival,
            x='Shock Count',
            y='Survival Rate',
            title="Survival Rate by Shock Count",
            labels={'Survival Rate': 'Discharge Survival Rate'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # CPR Duration
    st.subheader("💓 CPR Duration Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Median CPR Duration", f"{df['cpr_duration_min'].median():.1f} min")
    with col2:
        st.metric("Mean CPR Duration", f"{df['cpr_duration_min'].mean():.1f} min")
    with col3:
        q1, q3 = df['cpr_duration_min'].quantile(0.25), df['cpr_duration_min'].quantile(0.75)
        st.metric("IQR", f"{q1:.1f} - {q3:.1f} min")
    with col4:
        outliers = (df['cpr_duration_min'] > 60).sum()
        st.metric("Outliers (>60 min)", f"{outliers}")
    
    # CPR Duration vs Survival
    col1, col2 = st.columns(2)
    
    with col1:
        # Boxplot
        df_cpr_survival = df.copy()
        df_cpr_survival['Survival Status'] = df_cpr_survival['survival_to_discharge'].map({1: 'Survived', 0: 'Died'})
        fig = px.box(
            df_cpr_survival,
            x='Survival Status',
            y='cpr_duration_min',
            title="CPR Duration by Survival Status",
            labels={'cpr_duration_min': 'CPR Duration (minutes)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot
        fig = px.scatter(
            df,
            x='cpr_duration_min',
            y='rosc',
            color='survival_to_discharge',
            title="CPR Duration vs ROSC",
            labels={'cpr_duration_min': 'CPR Duration (minutes)', 'rosc': 'ROSC'},
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # CPR Duration Statistics by Survival
    st.subheader("📊 CPR Duration Statistics by Outcome")
    
    survived_cpr = df[df['survival_to_discharge']==1]['cpr_duration_min']
    died_cpr = df[df['survival_to_discharge']==0]['cpr_duration_min']
    
    cpr_stats = pd.DataFrame({
        'Outcome': ['Survived', 'Died'],
        'Median CPR (min)': [
            survived_cpr.median() if len(survived_cpr) > 0 else 0,
            died_cpr.median() if len(died_cpr) > 0 else 0
        ],
        'Mean CPR (min)': [
            survived_cpr.mean() if len(survived_cpr) > 0 else 0,
            died_cpr.mean() if len(died_cpr) > 0 else 0
        ],
        'IQR (min)': [
            f"{survived_cpr.quantile(0.25):.1f} - {survived_cpr.quantile(0.75):.1f}" if len(survived_cpr) > 0 else "N/A",
            f"{died_cpr.quantile(0.25):.1f} - {died_cpr.quantile(0.75):.1f}" if len(died_cpr) > 0 else "N/A"
        ]
    })
    st.dataframe(cpr_stats, use_container_width=True)
    
    # Arrest to CPR Delay
    st.subheader("⏱️ Arrest to CPR Delay Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Median Delay", f"{df['arrest_to_cpr_min'].median():.1f} min")
    with col2:
        delayed_cpr = (df['arrest_to_cpr_min'] > 2).sum()
        st.metric("Delayed CPR (>2 min)", f"{delayed_cpr} ({delayed_cpr/len(df)*100:.1f}%)")
    with col3:
        st.metric("Mean Delay", f"{df['arrest_to_cpr_min'].mean():.1f} min")
    
    # Delay distribution and impact
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df,
            x='arrest_to_cpr_min',
            nbins=30,
            title="Arrest-to-CPR Delay Distribution",
            labels={'arrest_to_cpr_min': 'Delay (minutes)', 'count': 'Frequency'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Delay vs survival
        delay_bins = pd.cut(df['arrest_to_cpr_min'], bins=[0, 1, 2, 5, 10, 100], labels=['<1', '1-2', '2-5', '5-10', '>10'])
        delay_survival = df.groupby(delay_bins)['survival_to_discharge'].mean().reset_index()
        delay_survival.columns = ['Delay (min)', 'Survival Rate']
        fig = px.bar(
            delay_survival,
            x='Delay (min)',
            y='Survival Rate',
            title="Survival Rate by Arrest-to-CPR Delay",
            labels={'Survival Rate': 'Discharge Survival Rate'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# 5. Outcomes
elif page == "✅ 5. Outcomes":
    st.title("✅ Outcomes Analysis")
    st.markdown("### Goal: Evaluate performance")
    
    # Overall outcomes
    st.subheader("📊 Overall Outcomes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rosc_rate = df['rosc'].mean()
        st.metric("ROSC Rate", f"{rosc_rate*100:.1f}%", f"{df['rosc'].sum()} cases")
    with col2:
        survival_24h_rate = df['survival_24h'].mean()
        st.metric("24h Survival Rate", f"{survival_24h_rate*100:.1f}%", f"{df['survival_24h'].sum()} cases")
    with col3:
        discharge_rate = df['survival_to_discharge'].mean()
        st.metric("Discharge Survival Rate", f"{discharge_rate*100:.1f}%", f"{df['survival_to_discharge'].sum()} cases")
    
    # Outcome flow
    st.subheader("📈 Outcome Flow")
    
    total = len(df)
    rosc_count = df['rosc'].sum()
    survival_24h_count = df['survival_24h'].sum()
    discharge_count = df['survival_to_discharge'].sum()
    
    fig = go.Figure(go.Sankey(
        node=dict(
            label=["Total Cases", "ROSC", "No ROSC", "24h Survival", "No 24h Survival", "Discharge Survival", "No Discharge"],
            color=["lightblue", "lightgreen", "lightcoral", "lightgreen", "lightcoral", "lightgreen", "lightcoral"]
        ),
        link=dict(
            source=[0, 0, 1, 1, 2, 3, 3],
            target=[1, 2, 3, 4, 4, 5, 6],
            value=[rosc_count, total-rosc_count, survival_24h_count, rosc_count-survival_24h_count, total-rosc_count, discharge_count, survival_24h_count-discharge_count]
        )
    ))
    fig.update_layout(title="Outcome Flow Diagram", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Outcomes by different factors
    st.subheader("🔍 Outcomes by Key Factors")
    
    # By Age
    st.markdown("#### By Age Group")
    age_bins = pd.cut(df['age'], bins=[0, 50, 65, 80, 100], labels=['<50', '50-65', '65-80', '>80'])
    age_outcomes = df.groupby(age_bins).agg({
        'rosc': 'mean',
        'survival_24h': 'mean',
        'survival_to_discharge': 'mean'
    }).reset_index()
    age_outcomes.columns = ['Age Group', 'ROSC', '24h Survival', 'Discharge Survival']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='ROSC', x=age_outcomes['Age Group'], y=age_outcomes['ROSC']*100))
    fig.add_trace(go.Bar(name='24h Survival', x=age_outcomes['Age Group'], y=age_outcomes['24h Survival']*100))
    fig.add_trace(go.Bar(name='Discharge Survival', x=age_outcomes['Age Group'], y=age_outcomes['Discharge Survival']*100))
    fig.update_layout(
        title="Outcome Rates by Age Group",
        yaxis_title="Rate (%)",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # By Comorbidity
    st.markdown("#### By Comorbidity")
    comorbidity_cols = ['hypertension', 'diabetes', 'coronary_artery_disease', 'heart_failure', 'copd']
    comorbidity_names = ['Hypertension', 'Diabetes', 'CAD', 'Heart Failure', 'COPD']
    
    comorb_outcomes = []
    for col, name in zip(comorbidity_cols, comorbidity_names):
        with_comorb = df[df[col]==1]
        without_comorb = df[df[col]==0]
        if len(with_comorb) > 0 and len(without_comorb) > 0:
            comorb_outcomes.append({
                'Comorbidity': name,
                'With - ROSC': with_comorb['rosc'].mean(),
                'With - Discharge': with_comorb['survival_to_discharge'].mean(),
                'Without - ROSC': without_comorb['rosc'].mean(),
                'Without - Discharge': without_comorb['survival_to_discharge'].mean()
            })
    
    df_comorb_outcomes = pd.DataFrame(comorb_outcomes)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='With - ROSC', x=df_comorb_outcomes['Comorbidity'], y=df_comorb_outcomes['With - ROSC']*100))
    fig.add_trace(go.Bar(name='With - Discharge', x=df_comorb_outcomes['Comorbidity'], y=df_comorb_outcomes['With - Discharge']*100))
    fig.add_trace(go.Bar(name='Without - ROSC', x=df_comorb_outcomes['Comorbidity'], y=df_comorb_outcomes['Without - ROSC']*100))
    fig.add_trace(go.Bar(name='Without - Discharge', x=df_comorb_outcomes['Comorbidity'], y=df_comorb_outcomes['Without - Discharge']*100))
    fig.update_layout(
        title="Outcome Rates by Comorbidity",
        yaxis_title="Rate (%)",
        xaxis_tickangle=-45,
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # By Location
    st.markdown("#### By Location (Top 5)")
    location_cols = [col for col in df.columns if col.startswith('loc_')]
    location_counts = {col: df[col].sum() for col in location_cols}
    top_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    loc_outcomes = []
    for col, count in top_locations:
        loc_df = df[df[col]==1]
        if len(loc_df) > 0:
            loc_outcomes.append({
                'Location': get_location_name(col),
                'ROSC': loc_df['rosc'].mean(),
                '24h Survival': loc_df['survival_24h'].mean(),
                'Discharge Survival': loc_df['survival_to_discharge'].mean()
            })
    
    df_loc_outcomes = pd.DataFrame(loc_outcomes)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='ROSC', x=df_loc_outcomes['Location'], y=df_loc_outcomes['ROSC']*100))
    fig.add_trace(go.Bar(name='24h Survival', x=df_loc_outcomes['Location'], y=df_loc_outcomes['24h Survival']*100))
    fig.add_trace(go.Bar(name='Discharge Survival', x=df_loc_outcomes['Location'], y=df_loc_outcomes['Discharge Survival']*100))
    fig.update_layout(
        title="Outcome Rates by Location",
        yaxis_title="Rate (%)",
        xaxis_tickangle=-45,
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # By Rhythm
    st.markdown("#### By Rhythm")
    rhythm_cols = [col for col in df.columns if col.startswith('rhythm_')]
    
    rhythm_outcomes = []
    for col in rhythm_cols:
        rhythm_df = df[df[col]==1]
        if len(rhythm_df) > 0:
            rhythm_outcomes.append({
                'Rhythm': get_rhythm_name(col),
                'ROSC': rhythm_df['rosc'].mean(),
                '24h Survival': rhythm_df['survival_24h'].mean(),
                'Discharge Survival': rhythm_df['survival_to_discharge'].mean(),
                'Cases': len(rhythm_df)
            })
    
    df_rhythm_outcomes = pd.DataFrame(rhythm_outcomes)
    df_rhythm_outcomes = df_rhythm_outcomes.sort_values('Discharge Survival', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='ROSC', x=df_rhythm_outcomes['Rhythm'], y=df_rhythm_outcomes['ROSC']*100))
    fig.add_trace(go.Bar(name='24h Survival', x=df_rhythm_outcomes['Rhythm'], y=df_rhythm_outcomes['24h Survival']*100))
    fig.add_trace(go.Bar(name='Discharge Survival', x=df_rhythm_outcomes['Rhythm'], y=df_rhythm_outcomes['Discharge Survival']*100))
    fig.update_layout(
        title="Outcome Rates by Rhythm",
        yaxis_title="Rate (%)",
        xaxis_tickangle=-45,
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # By CPR Duration
    st.markdown("#### By CPR Duration")
    cpr_bins = pd.cut(df['cpr_duration_min'], bins=[0, 10, 20, 30, 60, 120], labels=['<10', '10-20', '20-30', '30-60', '>60'])
    cpr_outcomes = df.groupby(cpr_bins).agg({
        'rosc': 'mean',
        'survival_24h': 'mean',
        'survival_to_discharge': 'mean'
    }).reset_index()
    cpr_outcomes.columns = ['CPR Duration (min)', 'ROSC', '24h Survival', 'Discharge Survival']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='ROSC', x=cpr_outcomes['CPR Duration (min)'], y=cpr_outcomes['ROSC']*100))
    fig.add_trace(go.Bar(name='24h Survival', x=cpr_outcomes['CPR Duration (min)'], y=cpr_outcomes['24h Survival']*100))
    fig.add_trace(go.Bar(name='Discharge Survival', x=cpr_outcomes['CPR Duration (min)'], y=cpr_outcomes['Discharge Survival']*100))
    fig.update_layout(
        title="Outcome Rates by CPR Duration",
        yaxis_title="Rate (%)",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# 6. Temporal Patterns
elif page == "⏱️ 6. Temporal Patterns":
    st.title("⏱️ Temporal Patterns Analysis")
    st.markdown("### Goal: Identify delays and timing issues")
    
    # CPR Duration Analysis
    st.subheader("💓 CPR Duration Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution
        fig = px.histogram(
            df,
            x='cpr_duration_min',
            nbins=50,
            title="CPR Duration Distribution",
            labels={'cpr_duration_min': 'CPR Duration (minutes)', 'count': 'Frequency'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Boxplot by outcome
        df_cpr_outcome = df.copy()
        df_cpr_outcome['Outcome'] = df_cpr_outcome['survival_to_discharge'].map({1: 'Survived', 0: 'Died'})
        fig = px.box(
            df_cpr_outcome,
            x='Outcome',
            y='cpr_duration_min',
            title="CPR Duration by Outcome",
            labels={'cpr_duration_min': 'CPR Duration (minutes)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # CPR Duration Statistics
    st.subheader("📊 CPR Duration Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Mean", f"{df['cpr_duration_min'].mean():.1f} min")
    with col2:
        st.metric("Median", f"{df['cpr_duration_min'].median():.1f} min")
    with col3:
        q1, q3 = df['cpr_duration_min'].quantile(0.25), df['cpr_duration_min'].quantile(0.75)
        st.metric("IQR", f"{q1:.1f} - {q3:.1f} min")
    with col4:
        outliers_60 = (df['cpr_duration_min'] > 60).sum()
        st.metric("Outliers (>60 min)", f"{outliers_60}")
    with col5:
        outliers_30 = (df['cpr_duration_min'] > 30).sum()
        st.metric("Long CPR (>30 min)", f"{outliers_30} ({outliers_30/len(df)*100:.1f}%)")
    
    # CPR Duration vs Outcomes
    st.subheader("📈 CPR Duration vs Outcomes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter: CPR duration vs ROSC
        fig = px.scatter(
            df,
            x='cpr_duration_min',
            y='rosc',
            color='survival_to_discharge',
            title="CPR Duration vs ROSC",
            labels={'cpr_duration_min': 'CPR Duration (minutes)', 'rosc': 'ROSC'},
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Survival rate by CPR duration bins
        cpr_bins = pd.cut(df['cpr_duration_min'], bins=[0, 5, 10, 15, 20, 30, 60, 120], 
                         labels=['<5', '5-10', '10-15', '15-20', '20-30', '30-60', '>60'])
        cpr_survival = df.groupby(cpr_bins)['survival_to_discharge'].mean().reset_index()
        cpr_survival.columns = ['CPR Duration (min)', 'Survival Rate']
        fig = px.bar(
            cpr_survival,
            x='CPR Duration (min)',
            y='Survival Rate',
            title="Survival Rate by CPR Duration",
            labels={'Survival Rate': 'Discharge Survival Rate'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Arrest-to-CPR Delay Analysis
    st.subheader("⏱️ Arrest-to-CPR Delay Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution
        fig = px.histogram(
            df,
            x='arrest_to_cpr_min',
            nbins=30,
            title="Arrest-to-CPR Delay Distribution",
            labels={'arrest_to_cpr_min': 'Delay (minutes)', 'count': 'Frequency'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Delay vs survival
        delay_bins = pd.cut(df['arrest_to_cpr_min'], bins=[0, 1, 2, 5, 10, 100], 
                           labels=['<1', '1-2', '2-5', '5-10', '>10'])
        delay_survival = df.groupby(delay_bins).agg({
            'survival_to_discharge': 'mean',
            'rosc': 'mean'
        }).reset_index()
        delay_survival.columns = ['Delay (min)', 'Discharge Survival', 'ROSC']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Discharge Survival', x=delay_survival['Delay (min)'], y=delay_survival['Discharge Survival']*100))
        fig.add_trace(go.Bar(name='ROSC', x=delay_survival['Delay (min)'], y=delay_survival['ROSC']*100))
        fig.update_layout(
            title="Outcome Rates by Arrest-to-CPR Delay",
            yaxis_title="Rate (%)",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Delay Statistics
    st.subheader("📊 Delay Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Delay", f"{df['arrest_to_cpr_min'].mean():.1f} min")
    with col2:
        st.metric("Median Delay", f"{df['arrest_to_cpr_min'].median():.1f} min")
    with col3:
        delayed_2min = (df['arrest_to_cpr_min'] > 2).sum()
        st.metric("Delayed (>2 min)", f"{delayed_2min} ({delayed_2min/len(df)*100:.1f}%)")
    with col4:
        delayed_5min = (df['arrest_to_cpr_min'] > 5).sum()
        st.metric("Delayed (>5 min)", f"{delayed_5min} ({delayed_5min/len(df)*100:.1f}%)")
    
    # Correlation Analysis
    st.subheader("🔗 Temporal Correlations")
    
    # Scatter: Delay vs CPR duration vs Survival
    fig = px.scatter(
        df,
        x='arrest_to_cpr_min',
        y='cpr_duration_min',
        color='survival_to_discharge',
        size='shock_count',
        title="Arrest-to-CPR Delay vs CPR Duration (colored by survival, sized by shocks)",
        labels={
            'arrest_to_cpr_min': 'Arrest-to-CPR Delay (minutes)',
            'cpr_duration_min': 'CPR Duration (minutes)'
        },
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary Table
    st.subheader("📋 Summary Statistics")
    
    summary_stats = pd.DataFrame({
        'Metric': [
            'Normal CPR Range (IQR)',
            'CPR Duration > 60 min',
            'Median CPR - Survivors',
            'Median CPR - Non-Survivors',
            'Median Delay',
            'Delayed CPR (>2 min)',
            'Correlation: Delay vs Survival'
        ],
        'Value': [
            f"{df['cpr_duration_min'].quantile(0.25):.1f} - {df['cpr_duration_min'].quantile(0.75):.1f} min",
            f"{(df['cpr_duration_min'] > 60).sum()} cases",
            f"{df[df['survival_to_discharge']==1]['cpr_duration_min'].median():.1f} min",
            f"{df[df['survival_to_discharge']==0]['cpr_duration_min'].median():.1f} min",
            f"{df['arrest_to_cpr_min'].median():.1f} min",
            f"{(df['arrest_to_cpr_min'] > 2).sum()} cases ({(df['arrest_to_cpr_min'] > 2).sum()/len(df)*100:.1f}%)",
            f"{df['arrest_to_cpr_min'].corr(df['survival_to_discharge']):.3f}"
        ]
    })
    st.dataframe(summary_stats, use_container_width=True, hide_index=True)

# 7. Correlation Analysis
elif page == "🔗 7. Correlation Analysis":
    st.title("🔗 Correlation Analysis")
    st.markdown("### Goal: What features are associated with survival?")
    
    # Prepare numeric columns for correlation
    numeric_cols = [
        'age', 'gender', 'smoking',
        'coronary_artery_disease', 'heart_failure', 'heart_disease',
        'hypertension', 'copd', 'diabetes', 'cancer', 'covid_on_admission',
        'shock_count', 'max_energy', 'shockable_rhythm',
        'cpr_duration_min', 'arrest_to_cpr_min'
    ]
    
    # Add location and rhythm dummies
    location_cols = [col for col in df.columns if col.startswith('loc_')]
    rhythm_cols = [col for col in df.columns if col.startswith('rhythm_')]
    
    all_corr_cols = numeric_cols + location_cols + rhythm_cols
    
    # Correlation with ROSC
    st.subheader("📊 Correlation with ROSC")
    
    rosc_corr = df[all_corr_cols + ['rosc']].corr()['rosc'].sort_values(ascending=False)
    rosc_corr = rosc_corr.drop('rosc')
    
    # Top correlations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Positive Correlations")
        top_positive = rosc_corr[rosc_corr > 0].head(10)
        fig = px.bar(
            x=top_positive.values,
            y=top_positive.index,
            orientation='h',
            title="Top 10 Features Correlated with ROSC (Positive)",
            labels={'x': 'Correlation Coefficient', 'y': 'Feature'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Top Negative Correlations")
        top_negative = rosc_corr[rosc_corr < 0].tail(10)
        fig = px.bar(
            x=top_negative.values,
            y=top_negative.index,
            orientation='h',
            title="Top 10 Features Correlated with ROSC (Negative)",
            labels={'x': 'Correlation Coefficient', 'y': 'Feature'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation with Discharge Survival
    st.subheader("📊 Correlation with Discharge Survival")
    
    survival_corr = df[all_corr_cols + ['survival_to_discharge']].corr()['survival_to_discharge'].sort_values(ascending=False)
    survival_corr = survival_corr.drop('survival_to_discharge')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Positive Correlations")
        top_positive_surv = survival_corr[survival_corr > 0].head(10)
        fig = px.bar(
            x=top_positive_surv.values,
            y=top_positive_surv.index,
            orientation='h',
            title="Top 10 Features Correlated with Discharge Survival (Positive)",
            labels={'x': 'Correlation Coefficient', 'y': 'Feature'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Top Negative Correlations")
        top_negative_surv = survival_corr[survival_corr < 0].tail(10)
        fig = px.bar(
            x=top_negative_surv.values,
            y=top_negative_surv.index,
            orientation='h',
            title="Top 10 Features Correlated with Discharge Survival (Negative)",
            labels={'x': 'Correlation Coefficient', 'y': 'Feature'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")
    
    # Select key features for heatmap
    key_features = [
        'age', 'gender', 'shockable_rhythm', 'shock_count',
        'cpr_duration_min', 'arrest_to_cpr_min',
        'hypertension', 'diabetes', 'coronary_artery_disease',
        'loc_ICU', 'rhythm_VF', 'rhythm_VT', 'rhythm_Asystole',
        'rosc', 'survival_24h', 'survival_to_discharge'
    ]
    
    corr_matrix = df[key_features].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Feature", y="Feature", color="Correlation"),
        x=key_features,
        y=key_features,
        color_continuous_scale='RdBu',
        aspect="auto",
        title="Correlation Heatmap of Key Features"
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Age Correlations")
        st.write(f"- **Age vs ROSC**: {df['age'].corr(df['rosc']):.3f}")
        st.write(f"- **Age vs 24h Survival**: {df['age'].corr(df['survival_24h']):.3f}")
        st.write(f"- **Age vs Discharge Survival**: {df['age'].corr(df['survival_to_discharge']):.3f}")
        
        st.markdown("#### Comorbidity Correlations")
        comorb_cols = ['hypertension', 'diabetes', 'coronary_artery_disease', 'heart_failure']
        for col in comorb_cols:
            corr = df[col].corr(df['survival_to_discharge'])
            st.write(f"- **{col.replace('_', ' ').title()} vs Survival**: {corr:.3f}")
    
    with col2:
        st.markdown("#### Intervention Correlations")
        st.write(f"- **Shockable Rhythm vs Survival**: {df['shockable_rhythm'].corr(df['survival_to_discharge']):.3f}")
        st.write(f"- **Shock Count vs Survival**: {df['shock_count'].corr(df['survival_to_discharge']):.3f}")
        st.write(f"- **CPR Duration vs Survival**: {df['cpr_duration_min'].corr(df['survival_to_discharge']):.3f}")
        st.write(f"- **Delay vs Survival**: {df['arrest_to_cpr_min'].corr(df['survival_to_discharge']):.3f}")
        
        st.markdown("#### Location Correlations")
        st.write(f"- **ICU vs Survival**: {df['loc_ICU'].corr(df['survival_to_discharge']):.3f}")
        if 'loc_OPERATING_ROOM' in df.columns:
            st.write(f"- **Operating Room vs Survival**: {df['loc_OPERATING_ROOM'].corr(df['survival_to_discharge']):.3f}")

# 8. Rare Events
elif page == "🔍 8. Rare Events":
    st.title("🔍 Rare Events Analysis")
    st.markdown("### Goal: Identify patterns in small subgroups")
    
    # Shockable Rhythms
    st.subheader("⚡ Shockable Rhythm Analysis")
    
    shockable_count = df['shockable_rhythm'].sum()
    shockable_rate = df['shockable_rhythm'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Shockable Cases", f"{shockable_count}")
    with col2:
        st.metric("Shockable Rate", f"{shockable_rate*100:.2f}%")
    with col3:
        if shockable_count > 0:
            shockable_survival = df[df['shockable_rhythm']==1]['survival_to_discharge'].mean()
            st.metric("Shockable Survival", f"{shockable_survival*100:.1f}%")
        else:
            st.metric("Shockable Survival", "N/A")
    with col4:
        non_shockable_survival = df[df['shockable_rhythm']==0]['survival_to_discharge'].mean()
        st.metric("Non-Shockable Survival", f"{non_shockable_survival*100:.1f}%")
    
    if shockable_count > 0:
        # VF/VT cases analysis
        shockable_df = df[df['shockable_rhythm']==1]
        
        st.markdown("#### Shockable Rhythm Cases Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # VF vs VT
            vf_cases = shockable_df['rhythm_VF'].sum()
            vt_cases = shockable_df['rhythm_VT'].sum()
            fig = px.pie(
                values=[vf_cases, vt_cases],
                names=['VF', 'VT'],
                title="VF vs VT Distribution"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Shockable cases by location
            location_cols = [col for col in df.columns if col.startswith('loc_')]
            shockable_by_loc = []
            for col in location_cols:
                count = len(shockable_df[shockable_df[col]==1])
                if count > 0:
                    shockable_by_loc.append({
                        'Location': get_location_name(col),
                        'Count': count
                    })
            
            if shockable_by_loc:
                df_shockable_loc = pd.DataFrame(shockable_by_loc)
                df_shockable_loc = df_shockable_loc.sort_values('Count', ascending=False)
                fig = px.bar(
                    df_shockable_loc,
                    x='Location',
                    y='Count',
                    title="Shockable Cases by Location"
                )
                fig.update_layout(height=300, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Shockable cases characteristics
        st.markdown("#### Shockable Cases Characteristics")
        
        shockable_stats = pd.DataFrame({
            'Characteristic': [
                'Mean Age',
                'Male %',
                'Mean Shock Count',
                'Mean CPR Duration (min)',
                'ROSC Rate',
                '24h Survival Rate',
                'Discharge Survival Rate'
            ],
            'Value': [
                f"{shockable_df['age'].mean():.1f}",
                f"{(shockable_df['gender'] == 1).mean()*100:.1f}%",
                f"{shockable_df['shock_count'].mean():.1f}",
                f"{shockable_df['cpr_duration_min'].mean():.1f}",
                f"{shockable_df['rosc'].mean()*100:.1f}%",
                f"{shockable_df['survival_24h'].mean()*100:.1f}%",
                f"{shockable_df['survival_to_discharge'].mean()*100:.1f}%"
            ]
        })
        st.dataframe(shockable_stats, use_container_width=True, hide_index=True)
    
    # Rare Locations
    st.subheader("📍 Rare Locations Analysis")
    
    location_cols = [col for col in df.columns if col.startswith('loc_')]
    location_counts = {col: df[col].sum() for col in location_cols}
    
    # Identify rare locations (< 5% of cases)
    rare_locations = {loc: count for loc, count in location_counts.items() if count < len(df) * 0.05 and count > 0}
    
    if rare_locations:
        st.markdown("#### Locations with < 5% of Cases")
        
        rare_loc_data = []
        for loc, count in rare_locations.items():
            loc_df = df[df[loc]==1]
            if len(loc_df) > 0:
                rare_loc_data.append({
                    'Location': get_location_name(loc),
                    'Cases': count,
                    'Percentage': count/len(df)*100,
                    'ROSC Rate': loc_df['rosc'].mean(),
                    'Survival Rate': loc_df['survival_to_discharge'].mean()
                })
        
        df_rare_loc = pd.DataFrame(rare_loc_data)
        df_rare_loc = df_rare_loc.sort_values('Cases')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_rare_loc,
                x='Location',
                y='Cases',
                title="Rare Locations - Case Counts"
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df_rare_loc,
                x='Location',
                y='Survival Rate',
                title="Rare Locations - Survival Rates"
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_rare_loc, use_container_width=True, hide_index=True)
    else:
        st.info("No rare locations found (< 5% of cases).")
    
    # Rare Rhythms
    st.subheader("💓 Rare Rhythms Analysis")
    
    rhythm_cols = [col for col in df.columns if col.startswith('rhythm_')]
    rhythm_counts = {col: df[col].sum() for col in rhythm_cols}
    
    # Identify rare rhythms
    rare_rhythms = {rhythm: count for rhythm, count in rhythm_counts.items() if count < len(df) * 0.05 and count > 0}
    
    if rare_rhythms:
        st.markdown("#### Rhythms with < 5% of Cases")
        
        rare_rhythm_data = []
        for rhythm, count in rare_rhythms.items():
            rhythm_df = df[df[rhythm]==1]
            if len(rhythm_df) > 0:
                rare_rhythm_data.append({
                    'Rhythm': get_rhythm_name(rhythm),
                    'Cases': count,
                    'Percentage': count/len(df)*100,
                    'ROSC Rate': rhythm_df['rosc'].mean(),
                    'Survival Rate': rhythm_df['survival_to_discharge'].mean()
                })
        
        df_rare_rhythm = pd.DataFrame(rare_rhythm_data)
        df_rare_rhythm = df_rare_rhythm.sort_values('Cases')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_rare_rhythm,
                x='Rhythm',
                y='Cases',
                title="Rare Rhythms - Case Counts"
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df_rare_rhythm,
                x='Rhythm',
                y='Survival Rate',
                title="Rare Rhythms - Survival Rates"
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_rare_rhythm, use_container_width=True, hide_index=True)
    else:
        st.info("No rare rhythms found (< 5% of cases).")
    
    # Deaths by Location
    st.subheader("💀 Deaths by Location")
    
    deaths_by_loc = []
    for col in location_cols:
        loc_df = df[df[col]==1]
        if len(loc_df) > 0:
            deaths = (loc_df['survival_to_discharge'] == 0).sum()
            deaths_by_loc.append({
                'Location': get_location_name(col),
                'Total Cases': len(loc_df),
                'Deaths': deaths,
                'Death Rate': (deaths/len(loc_df))*100,
                'Survival Rate': loc_df['survival_to_discharge'].mean()*100
            })
    
    df_deaths_loc = pd.DataFrame(deaths_by_loc)
    df_deaths_loc = df_deaths_loc.sort_values('Death Rate', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            df_deaths_loc,
            x='Location',
            y='Deaths',
            title="Deaths by Location",
            labels={'Deaths': 'Number of Deaths'}
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            df_deaths_loc,
            x='Location',
            y='Death Rate',
            title="Death Rate by Location",
            labels={'Death Rate': 'Death Rate (%)'}
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(df_deaths_loc, use_container_width=True, hide_index=True)
    
    # Summary Table for Rare Events
    st.subheader("📋 Rare Events Summary")
    
    # Calculate rare location and rhythm counts safely
    rare_loc_count = sum(rare_locations.values()) if rare_locations else 0
    rare_rhythm_count = sum(rare_rhythms.values()) if rare_rhythms else 0
    
    rare_events_summary = pd.DataFrame({
        'Rare Event': [
            'Shockable Rhythm Cases',
            'Rare Location Cases (<5%)',
            'Rare Rhythm Cases (<5%)',
            'Long CPR (>60 min)',
            'Delayed CPR (>5 min)'
        ],
        'Count': [
            df['shockable_rhythm'].sum(),
            rare_loc_count,
            rare_rhythm_count,
            (df['cpr_duration_min'] > 60).sum(),
            (df['arrest_to_cpr_min'] > 5).sum()
        ],
        'Percentage': [
            f"{df['shockable_rhythm'].mean()*100:.2f}%",
            f"{rare_loc_count/len(df)*100:.2f}%",
            f"{rare_rhythm_count/len(df)*100:.2f}%",
            f"{(df['cpr_duration_min'] > 60).sum()/len(df)*100:.2f}%",
            f"{(df['arrest_to_cpr_min'] > 5).sum()/len(df)*100:.2f}%"
        ]
    })
    st.dataframe(rare_events_summary, use_container_width=True, hide_index=True)
