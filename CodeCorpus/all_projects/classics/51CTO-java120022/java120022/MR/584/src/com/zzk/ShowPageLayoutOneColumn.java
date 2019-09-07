package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class ShowPageLayoutOneColumn {
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\单列显示.pdf"));// 关联文档对象与输出流
            writer.setViewerPreferences(PdfWriter.PageLayoutOneColumn); // 设置阅读器单列显示
            document.open();// 打开文档
            document.add(new Paragraph("ShowOneColumn Page 1."));// 向文档中添加内容
            document.newPage();// 增加新页
            document.add(new Paragraph("ShowOneColumn Page 2."));// 向文档中添加内容
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
        
    }
}
