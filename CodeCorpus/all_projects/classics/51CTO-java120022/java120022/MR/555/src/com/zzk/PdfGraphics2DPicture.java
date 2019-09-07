package com.zzk;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import javax.imageio.ImageIO;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfGraphics2D;
import com.itextpdf.text.pdf.PdfWriter;

public class PdfGraphics2DPicture {
    
    public static void main(String[] args) {
        Document document = new Document(); // 打开文档
        try {
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("c:\\使用PdfGraphics2D绘制图片.pdf"));// 关联文档与输出流
            document.open();// 打开文档
            PdfContentByte cb = writer.getDirectContent();// 获取文档内容
            PdfGraphics2D g = (PdfGraphics2D) cb.createGraphics(700, 800);// 创建PdfGraphics2D对象
            BufferedImage image = ImageIO.read(new File("image/picture.jpg"));// 获取图片
            g.drawImage(image, 50, 10, null);// 绘制图片
            g.dispose();// 部署
            cb.stroke();// 确认绘制的内容
            document.close();// 关闭文档
        } catch (IOException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
    }
}
