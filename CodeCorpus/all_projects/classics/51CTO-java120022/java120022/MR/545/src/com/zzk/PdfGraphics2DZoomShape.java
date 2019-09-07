package com.zzk;

import java.awt.Color;
import java.awt.geom.Rectangle2D;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfGraphics2D;
import com.itextpdf.text.pdf.PdfWriter;

public class PdfGraphics2DZoomShape {
    
    public static void main(String[] args) throws MalformedURLException {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\使用PdfGraphics2D缩放绘制的图形.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            PdfContentByte cb = writer.getDirectContent();// 获取文档内容
            PdfGraphics2D g = (PdfGraphics2D) cb.createGraphics(700, 800);// 创建Graphics和坐标
            Rectangle2D rect = new Rectangle2D.Double(50, 30, 120, 150);// 创建原矩形对象
            g.setColor(Color.BLUE);// 设置颜色
            g.fill(rect);// 绘制有填充色的图形
            rect = new Rectangle2D.Double(150, 30, 120, 150);// 创建与原矩形大小相同的矩形对象
            g.scale(1.4, 1.2f);// 缩放矩形对象
            g.setColor(Color.PINK);// 设置颜色
            g.fill(rect);// 绘制有填充色的图形
            g.dispose();// 部署
            cb.stroke();// 确认绘制内容
            document.close();// 关闭文档
        } catch (DocumentException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
