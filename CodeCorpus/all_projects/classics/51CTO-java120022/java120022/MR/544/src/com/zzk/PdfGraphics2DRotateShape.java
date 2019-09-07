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

public class PdfGraphics2DRotateShape {
    
    public static void main(String[] args) throws MalformedURLException {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\使用PdfGraphics2D旋转绘制的图形.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            PdfContentByte cb = writer.getDirectContent();// 获取文档内容
            PdfGraphics2D g = (PdfGraphics2D) cb.createGraphics(700, 800);// 创建PdfGraphics2D对象
            Rectangle2D rect = new Rectangle2D.Double(200, 200, 150, 200);// 创建矩形对象
            g.setColor(Color.BLUE);// 设置图形颜色
            g.rotate(20, 380, 150);// 旋转图形
            g.draw(rect);// 绘制矩形对象
            g.dispose();// 部署
            document.close();// 关闭文档
        } catch (DocumentException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
