package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class CreateNewPage {
    
    public static void main(String[] args) {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\创建新页.pdf"));// 关联文档与输出流
            document.open();// 打开文档
            document.add(new Paragraph("Old Page"));// 为第一页添加内容
            document.newPage();// 创建新的页
            document.add(new Paragraph("New Page"));// 为新页添加内容
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
    }
}
