package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfWriter;

public class DrawRectangleDemo {
    public static void main(String[] args) {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\绘制矩形.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            PdfContentByte cb = writer.getDirectContent();// 获取内容
            cb.rectangle(50, 650, 200, 150); // 绘制矩形
            cb.stroke();// 确认绘制的矩形
            cb.rectangle(70, 675, 160, 100); // 绘制矩形
            cb.stroke();// 确认绘制的矩形
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
    }
}
