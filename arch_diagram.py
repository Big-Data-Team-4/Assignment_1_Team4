import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import RegularPolygon, Circle
import numpy as np

def create_compact_icon_diagram():
    """
    Creates a compact diagram with hexagonal icons like GCP reference architecture
    Updated with sequential processing pipeline and XBRL validation-only flow
    """
    
    # Figure size adjusted for new layout
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Professional blue color palette like GCP
    primary_blue = '#4285F4'    # Google Blue
    light_blue = '#E3F2FD'     # Light blue background
    dark_blue = '#1565C0'      # Dark blue
    gray_bg = '#F5F5F5'        # Light gray background
    
    # Title
    ax.text(5, 6.6, 'Project LANTERN', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#333333')
    ax.text(5, 6.3, 'Financial Document Processing Pipeline', ha='center', va='center', 
            fontsize=9, color='#666666', style='italic')
    
    def draw_hexagon_icon(center, size, color, icon_type, label, sublabel=""):
        """Draw hexagonal icon with specific icons inside"""
        x, y = center
        
        # Hexagon
        hexagon = RegularPolygon((x, y), 6, radius=size, 
                                facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(hexagon)
        
        # Icons inside hexagon
        if icon_type == 'file':
            # Document icon
            rect = plt.Rectangle((x-0.12, y-0.15), 0.24, 0.3, 
                               facecolor='white', edgecolor='none')
            ax.add_patch(rect)
            # Document lines
            for i in range(3):
                ax.plot([x-0.08, x+0.08], [y+0.05-i*0.08, y+0.05-i*0.08], 
                       color=color, linewidth=1.5)
        
        elif icon_type == 'gear':
            # Gear/settings icon
            circle = Circle((x, y), 0.08, facecolor='white', edgecolor='none')
            ax.add_patch(circle)
            # Gear teeth
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                tooth_x = x + 0.12 * np.cos(angle)
                tooth_y = y + 0.12 * np.sin(angle)
                small_rect = plt.Rectangle((tooth_x-0.02, tooth_y-0.02), 0.04, 0.04, 
                                         facecolor='white', edgecolor='none')
                ax.add_patch(small_rect)
        
        elif icon_type == 'processor':
            # CPU/Processor icon
            rect = plt.Rectangle((x-0.1, y-0.1), 0.2, 0.2, 
                               facecolor='white', edgecolor='none')
            ax.add_patch(rect)
            # Processor grid
            for i in range(3):
                for j in range(3):
                    small_rect = plt.Rectangle((x-0.06+i*0.04, y-0.06+j*0.04), 0.02, 0.02, 
                                             facecolor=color, edgecolor='none')
                    ax.add_patch(small_rect)
        
        elif icon_type == 'cloud':
            # Cloud icon (simplified)
            cloud_parts = [
                Circle((x-0.05, y), 0.06, facecolor='white', edgecolor='none'),
                Circle((x+0.05, y), 0.06, facecolor='white', edgecolor='none'),
                Circle((x, y+0.04), 0.08, facecolor='white', edgecolor='none'),
                Circle((x, y-0.02), 0.1, facecolor='white', edgecolor='none')
            ]
            for part in cloud_parts:
                ax.add_patch(part)
        
        elif icon_type == 'transform':
            # Transformation arrows
            ax.arrow(x-0.08, y, 0.06, 0, head_width=0.03, head_length=0.02, 
                    fc='white', ec='none')
            ax.arrow(x+0.02, y+0.05, -0.06, 0, head_width=0.03, head_length=0.02, 
                    fc='white', ec='none')
        
        elif icon_type == 'check':
            # Checkmark
            ax.plot([x-0.06, x-0.02, x+0.06], [y, y-0.06, y+0.06], 
                   color='white', linewidth=3, solid_capstyle='round')
        
        # Label below hexagon
        ax.text(x, y-0.55, label, ha='center', va='center', 
                fontsize=8, fontweight='bold', color='#333333')
        if sublabel:
            ax.text(x, y-0.7, sublabel, ha='center', va='center', 
                    fontsize=7, color='#666666')
    
    # Layer 1: Data Sources (Top)
    draw_hexagon_icon((1.5, 5.5), 0.3, primary_blue, 'file', 'SEC PDFs', 'source data')
    draw_hexagon_icon((8.5, 5.5), 0.3, '#FFC107', 'file', 'XBRL Files', 'validation only')
    
    # Layer 2: DVC Controller (Only for SEC PDFs)
    draw_hexagon_icon((2.5, 4.5), 0.3, dark_blue, 'gear', 'DVC Controller', 'orchestration')
    
    # Layer 3: Processing Pipeline (Sequential flow)
    draw_hexagon_icon((1, 3.5), 0.25, primary_blue, 'processor', 'pdfplumber', 'processing')
    draw_hexagon_icon((2.5, 3.5), 0.25, primary_blue, 'processor', 'Camelot', 'processing')
    draw_hexagon_icon((4, 3.5), 0.25, primary_blue, 'processor', 'LayoutParser', 'processing')
    draw_hexagon_icon((5.5, 3.5), 0.25, primary_blue, 'processor', 'Docling', 'processing')
    draw_hexagon_icon((7.5, 3.5), 0.25, '#00B4D8', 'cloud', 'Azure AI', 'fallback')
    
    # Layer 4: Outputs (Individual outputs for each processor)
    draw_hexagon_icon((1, 2.5), 0.2, '#9C27B0', 'file', 'pdf_output', 'text files')
    draw_hexagon_icon((2.5, 2.5), 0.2, '#9C27B0', 'file', 'camelot_output', 'CSV tables')
    draw_hexagon_icon((4, 2.5), 0.2, '#9C27B0', 'file', 'layout_output', 'structured')
    draw_hexagon_icon((5.5, 2.5), 0.2, '#9C27B0', 'file', 'docling_output', 'formatted')
    
    # Layer 5: Integration (Bottom row)
    draw_hexagon_icon((3, 1.5), 0.3, '#7C3AED', 'transform', 'Format\nConverter', 'MD•JSON•JSONL')
    draw_hexagon_icon((7, 1.5), 0.3, '#EA580C', 'check', 'XBRL\nValidator', 'cross-verification')
    
    # Clean arrows - minimal style
    def draw_minimal_arrow(start, end, color='#666666', alpha=0.6, style='-'):
        """Draw arrow with optional dashed style for fallback connections"""
        linestyle = '--' if style == 'dashed' else '-'
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color=color, 
                                 lw=1.5, alpha=alpha, linestyle=linestyle))
    
    # Main flow arrows - Sequential processing pipeline
    draw_minimal_arrow((1.5, 5.2), (2.2, 4.8))  # SEC PDFs to DVC
    draw_minimal_arrow((2.5, 4.2), (2.5, 3.8))  # DVC to processing start
    
    # Sequential processing arrows (left to right)
    draw_minimal_arrow((1.25, 3.5), (2.25, 3.5))     # pdfplumber -> Camelot
    draw_minimal_arrow((2.75, 3.5), (3.75, 3.5))     # Camelot -> LayoutParser
    draw_minimal_arrow((4.25, 3.5), (5.25, 3.5))     # LayoutParser -> Docling
    draw_minimal_arrow((5.75, 3.5), (7.25, 3.5), alpha=0.4, style='dashed')  # Docling -> Azure (fallback)
    
    # Processing to outputs (vertical arrows)
    draw_minimal_arrow((1, 3.25), (1, 2.75))      # pdfplumber to output
    draw_minimal_arrow((2.5, 3.25), (2.5, 2.75))  # Camelot to output
    draw_minimal_arrow((4, 3.25), (4, 2.75))      # LayoutParser to output
    draw_minimal_arrow((5.5, 3.25), (5.5, 2.75))  # Docling to output
    
    # Outputs to integration
    draw_minimal_arrow((1, 2.25), (2.7, 1.8))     # pdfplumber output to Format Converter
    draw_minimal_arrow((2.5, 2.25), (2.8, 1.8))   # Camelot output to Format Converter
    draw_minimal_arrow((4, 2.25), (3.2, 1.8))     # LayoutParser output to Format Converter
    
    # XBRL validation (separate from DVC pipeline)
    draw_minimal_arrow((8.5, 5.2), (7.3, 1.8), '#FFC107')  # XBRL directly to validator
    draw_minimal_arrow((5.5, 2.25), (6.7, 1.8))   # Docling output to validator
    
    # Background sections with subtle colors
    # Data sources section (top)
    data_bg = plt.Rectangle((0.5, 5), 9, 1, facecolor=light_blue, alpha=0.3, zorder=0)
    ax.add_patch(data_bg)
    ax.text(0.7, 5.3, 'Data Sources', fontsize=8, color='#666666', style='italic')
    
    # DVC Controller section
    dvc_bg = plt.Rectangle((1.5, 4.2), 2, 0.6, facecolor='#E1F5FE', alpha=0.5, zorder=0)
    ax.add_patch(dvc_bg)
    ax.text(0.7, 4.5, 'Controller', fontsize=8, color='#666666', style='italic')
    
    # Processing pipeline section  
    proc_bg = plt.Rectangle((0.5, 3.2), 8, 0.6, facecolor='#FFF3E0', alpha=0.5, zorder=0)
    ax.add_patch(proc_bg)
    ax.text(0.7, 3.7, 'Processing Pipeline', fontsize=8, color='#666666', style='italic')
    
    # Outputs section
    output_bg = plt.Rectangle((0.5, 2.2), 6, 0.6, facecolor='#F3E5F5', alpha=0.5, zorder=0)
    ax.add_patch(output_bg)
    ax.text(0.7, 2.6, 'Outputs', fontsize=8, color='#666666', style='italic')
    
    # Integration section
    integ_bg = plt.Rectangle((0.5, 1.2), 9, 0.6, facecolor='#E8F5E8', alpha=0.5, zorder=0)
    ax.add_patch(integ_bg)
    ax.text(0.7, 1.7, 'Integration & Validation', fontsize=8, color='#666666', style='italic')
    
    # Bottom title bar like GCP example
    title_bar = plt.Rectangle((0, 0.2), 10, 0.4, facecolor='#424242', edgecolor='none')
    ax.add_patch(title_bar)
    ax.text(5, 0.4, 'AI-Powered Financial Document Processing', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('project_lantern_updated_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.05)
    plt.show()
    print("Updated architecture diagram saved as 'project_lantern_updated_architecture.png'")

# Alternative ultra-compact version with updates
def create_ultra_compact_version():
    """
    Even more compact version with updated flow
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    primary_blue = '#4285F4'
    
    # Simple title
    ax.text(4, 4.7, 'Project LANTERN Architecture', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#333333')
    
    def simple_hex_icon(center, color, label):
        """Simplified hexagon with label"""
        x, y = center
        hexagon = RegularPolygon((x, y), 6, radius=0.2, 
                                facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(hexagon)
        ax.text(x, y-0.4, label, ha='center', va='center', 
                fontsize=7, fontweight='bold', color='#333333')
    
    # Compact layout
    # Top: Inputs
    simple_hex_icon((1.5, 4.2), primary_blue, 'SEC\nPDFs')
    simple_hex_icon((6.5, 4.2), '#FFC107', 'XBRL\nFiles')
    
    # DVC Controller
    simple_hex_icon((1.5, 3.4), '#1565C0', 'DVC')
    
    # Processing row (sequential)
    processors = [
        ('pdf', 0.8), ('Camelot', 2), ('Layout', 3.2), ('Docling', 4.4), ('Azure', 5.6)
    ]
    for name, x in processors:
        color = '#00B4D8' if name == 'Azure' else primary_blue
        simple_hex_icon((x, 2.6), color, name)
    
    # Outputs
    for name, x in processors[:-1]:  # Exclude Azure from outputs
        simple_hex_icon((x, 1.8), '#9C27B0', f'{name}\nout')
    
    # Integration
    simple_hex_icon((2.5, 1), '#7C3AED', 'Format\nConvert')
    simple_hex_icon((5.5, 1), '#EA580C', 'XBRL\nValidate')
    
    # Sequential arrows
    ax.plot([0.8, 5.6], [2.6, 2.6], color='#CCCCCC', linewidth=1.5, alpha=0.5)
    ax.plot([4.6, 5.4], [2.6, 2.6], color='#CCCCCC', linewidth=1.5, alpha=0.3, linestyle='--')  # Fallback
    
    # Vertical arrows to outputs
    for _, x in processors[:-1]:
        ax.plot([x, x], [2.4, 2], color='#666666', linewidth=1, alpha=0.7)
    
    # XBRL direct to validator
    ax.plot([6.5, 5.5], [4, 1.2], color='#FFC107', linewidth=1.5, alpha=0.7)
    
    # Bottom bar
    bar = plt.Rectangle((0, 0.5), 8, 0.3, facecolor='#424242')
    ax.add_patch(bar)
    ax.text(4, 0.65, 'Sequential Processing with XBRL Validation', ha='center', va='center', 
            fontsize=9, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('project_lantern_ultra_compact_updated.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.05)
    plt.show()
    print("Ultra compact updated diagram saved as 'project_lantern_ultra_compact_updated.png'")

if __name__ == "__main__":
    print("Creating updated Project LANTERN architecture diagrams...")
    
    # Main compact version with detailed icons and updated flow
    create_compact_icon_diagram()
    
    # Ultra compact version with updates
    create_ultra_compact_version()
    
    print("Updated architecture diagrams created successfully!")
    print("\nKey changes implemented:")
    print("1. XBRL files shown as validation-only (yellow) - bypasses DVC")
    print("2. Sequential processing arrows: pdfplumber → Camelot → LayoutParser → Docling")
    print("3. Individual output hexagons for each processor")
    print("4. Azure shown as fallback with dashed arrow")
    print("5. Clear separation of pipeline stages with background sections")