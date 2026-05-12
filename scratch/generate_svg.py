import svgwrite

def generate_academic_svg(filename):
    dwg = svgwrite.Drawing(filename, profile='tiny', size=(1000, 400))
    
    # 定义颜色和字体
    bg_color = "#FFFFFF"
    box_color = "#F0F4F8"
    stroke_color = "#2D3748"
    text_color = "#1A202C"
    highlight_color = "#EBF8FF"
    
    # 1. 第一阶段：知识表示层
    dwg.add(dwg.rect(insert=(10, 50), size=(280, 300), rx=10, ry=10, fill=box_color, stroke=stroke_color, stroke_width=2))
    dwg.add(dwg.text("第一阶段：知识表示层", insert=(30, 80), fill=text_color, font_size="16px", font_weight="bold"))
    dwg.add(dwg.text("PDF 解析 -> OCR -> MTFC 分块", insert=(30, 110), fill=text_color, font_size="12px"))
    
    # E, F, V 标签
    tags = ["E: 引导解释", "F: 数学公式", "V: 变量定义", "E: 延伸说明"]
    for i, tag in enumerate(tags):
        dwg.add(dwg.rect(insert=(40, 140 + i*40), size=(220, 30), fill="white", stroke=stroke_color, stroke_width=1))
        dwg.add(dwg.text(tag, insert=(50, 160 + i*40), fill=text_color, font_size="12px"))

    # 箭头 1
    dwg.add(dwg.line(start=(290, 200), end=(340, 200), stroke=stroke_color, stroke_width=2))
    dwg.add(dwg.polyline(points=[(330, 195), (340, 200), (330, 205)], fill="none", stroke=stroke_color, stroke_width=2))

    # 2. 第二阶段：语义逻辑链接 (核心)
    dwg.add(dwg.rect(insert=(350, 50), size=(280, 300), rx=10, ry=10, fill=highlight_color, stroke="#3182CE", stroke_width=3))
    dwg.add(dwg.text("第二阶段：语义逻辑链接", insert=(370, 80), fill="#2C5282", font_size="16px", font_weight="bold"))
    dwg.add(dwg.text("三明治结构耦合 (E-F-V-E)", insert=(370, 110), fill="#2C5282", font_size="12px"))
    
    # 逻辑单元展示
    dwg.add(dwg.rect(insert=(380, 150), size=(220, 150), fill="white", stroke="#3182CE", stroke_dasharray="5,5"))
    dwg.add(dwg.text("STEM 逻辑单元 (Logic Unit)", insert=(390, 175), fill="#2C5282", font_size="12px", font_weight="bold"))
    dwg.add(dwg.text("{ E + F + V + E }", insert=(430, 220), fill="#2C5282", font_size="18px"))

    # 箭头 2
    dwg.add(dwg.line(start=(630, 200), end=(680, 200), stroke=stroke_color, stroke_width=2))
    dwg.add(dwg.polyline(points=[(670, 195), (680, 200), (670, 205)], fill="none", stroke=stroke_color, stroke_width=2))

    # 3. 第三阶段：逻辑约束推理
    dwg.add(dwg.rect(insert=(690, 50), size=(280, 300), rx=10, ry=10, fill=box_color, stroke=stroke_color, stroke_width=2))
    dwg.add(dwg.text("第三阶段：逻辑约束推理", insert=(710, 80), fill=text_color, font_size="16px", font_weight="bold"))
    
    # LLM 与 过滤器
    dwg.add(dwg.circle(center=(830, 200), r=40, fill="white", stroke=stroke_color, stroke_width=2))
    dwg.add(dwg.text("LLM", insert=(815, 205), fill=text_color, font_size="14px", font_weight="bold"))
    
    dwg.add(dwg.rect(insert=(730, 260), size=(200, 40), fill="#FED7D7", stroke="#E53E3E", stroke_width=1))
    dwg.add(dwg.text("逻辑优先约束 (Policy)", insert=(745, 285), fill="#C53030", font_size="12px"))
    
    # 两个输入箭头
    dwg.add(dwg.line(start=(690, 200), end=(790, 200), stroke=stroke_color, stroke_width=2))
    dwg.add(dwg.line(start=(830, 260), end=(830, 240), stroke="#C53030", stroke_width=2))

    dwg.save()

if __name__ == "__main__":
    generate_academic_svg("st_mtfc_v2_architecture.svg")
    print("SVG 生成成功: st_mtfc_v2_architecture.svg")
