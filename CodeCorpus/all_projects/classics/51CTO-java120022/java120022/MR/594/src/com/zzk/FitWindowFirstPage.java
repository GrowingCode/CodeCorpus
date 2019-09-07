package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class FitWindowFirstPage {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\使文档窗口适合显示第一页.pdf"));// 关联文档对象与输出流
            writer.setViewerPreferences(PdfWriter.FitWindow);// 设置阅读器适合显示第一页
            document.open();// 打开文档
            for (int i = 1; i <= 100; i++) {
                document.add(new Paragraph("Fit show Page 1. row " + i));// 向文档中添加段落内容
            }
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
    }
}
