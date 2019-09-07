package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class ShowPageLayoutTwoColumnLeft {
    
    public static void main(String args[]) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter writer = PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\双列显示奇页在左.pdf"));// 关联文档对象与输出流
            writer.setViewerPreferences(PdfWriter.PageLayoutTwoColumnLeft);// 设置阅读器双列显示奇页在左
            document.open();// 打开文档
            document.add(new Paragraph("This is Odd Page 1"));// 向文档中添加内容
            document.newPage();// 新增第二页
            document.add(new Paragraph("this is Even Page 2"));// 向文档中添加内容
            document.newPage();// 新增第三页
            document.add(new Paragraph("This is Odd Page 3"));// 向文档中添加内容
            document.newPage();// 新增第四页
            document.add(new Paragraph("This is Even Page 4"));// 向文档中添加内容
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
    }
    
}
