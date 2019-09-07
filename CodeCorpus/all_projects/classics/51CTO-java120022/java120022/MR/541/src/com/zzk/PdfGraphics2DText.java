package com.zzk;

import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfGraphics2D;
import com.itextpdf.text.pdf.PdfWriter;

public class PdfGraphics2DText {
    
    public static void main(String[] args) throws MalformedURLException {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\使用PdfGraphics2D绘制文本.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            PdfContentByte cb = writer.getDirectContent();// 获取文档内容
            PdfGraphics2D g = (PdfGraphics2D) cb.createGraphics(700, 800);// 获得PdfGraphics2D对象
            g.drawString("draw text. ", 54, 10);// 绘制文本
            g.drawString("second row text. ", 54, 30);// 绘制文本
            g.drawString("third row text. ", 54, 50);// 绘制文本
            g.dispose();// 部署
            cb.stroke();// 确认绘制的内容
            document.close();// 关闭文档
        } catch (DocumentException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
