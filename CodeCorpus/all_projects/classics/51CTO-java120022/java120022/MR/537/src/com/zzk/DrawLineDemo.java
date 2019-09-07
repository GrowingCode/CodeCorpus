package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfWriter;

public class DrawLineDemo {
    
    public static void main(String[] args) {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\绘制直线.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            PdfContentByte cb = writer.getDirectContent();// 获取内容
            cb.moveTo(50, 780); // 绘制起点坐标
            cb.lineTo(260, 780); // 绘制终点坐标
            cb.stroke(); // 确认直线的绘制
            cb.moveTo(50, 750);
            cb.lineTo(260, 750);
            cb.stroke();
            cb.moveTo(50, 720);
            cb.lineTo(260, 720);
            cb.stroke();
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
    }
}
