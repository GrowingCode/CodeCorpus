package com.zzk;

import java.awt.Color;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfGraphics2D;
import com.itextpdf.text.pdf.PdfWriter;

public class PdfGraphics2DFillShape {
    public static void main(String[] args) throws MalformedURLException {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\使用PdfGraphics2D绘制有填充色的图形.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            PdfContentByte cb = writer.getDirectContent();// 获取文档内容
            PdfGraphics2D g = (PdfGraphics2D) cb.createGraphics(700, 800);// 创建Graphics和坐标
            Rectangle2D rect = new Rectangle2D.Double(120, 130, 200, 100);// 创建矩形对象
            Ellipse2D circle = new Ellipse2D.Double();// 创建椭圆对象
            circle.setFrameFromCenter(220, 80, 370, 120);// 设置椭圆形的中心点坐标和角点坐标
            g.setColor(Color.BLUE);// 设置颜色
            g.fill(rect);// 绘制填充的矩形
            g.setColor(Color.PINK);// 设置颜色
            g.fill(circle);// 绘制填充的圆形
            g.dispose();// 部署
            cb.stroke();// 确认绘制图形
            document.close();// 关闭文档
        } catch (DocumentException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
