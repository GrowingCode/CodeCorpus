package com.zzk;

import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfWriter;

public class UseGraphics2D {
    
    public static void main(String[] args) throws MalformedURLException {
        Document document = new Document(); // 创建文档对象
        try {
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\使用Graphics2D绘制图形.pdf"));// 关联文档对象与输出流
            document.open(); // 打开文档
            PdfContentByte cb = writer.getDirectContent(); // 获取文档内容
            Graphics2D g = cb.createGraphics(850, 850); // 创建Graphics和坐标
            Rectangle2D rect1 = new Rectangle2D.Double(50, 50, 200, 150); // 创建矩形对象
            g.draw(rect1); // 绘制矩形
            Rectangle2D rect2 = new Rectangle2D.Double(70, 70, 160, 110); // 创建矩形对象
            g.fill(rect2); // 绘制填充矩形
            g.dispose(); // 部署
            cb.stroke(); // 确认绘制的图形
            document.close(); // 关闭文档
        } catch (DocumentException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
}
