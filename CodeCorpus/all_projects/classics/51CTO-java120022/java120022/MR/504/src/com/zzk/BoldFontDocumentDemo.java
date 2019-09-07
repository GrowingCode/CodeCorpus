package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfWriter;


public class BoldFontDocumentDemo {
    
    /**
     * @param args
     */
    public static void main(String[] args) {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter.getInstance(document, new FileOutputStream("c:\\Java.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont bfChinese = BaseFont.createFont(BaseFont.HELVETICA, BaseFont.CP1252, BaseFont.NOT_EMBEDDED);// 定义基本字体
            Font contentFont = new Font(bfChinese, 12, Font.BOLD);// 定义加粗字体
            document.add(new Paragraph("www.mingribccd.com",contentFont));// 向文档中添加内容并指定加粗字体
            document.add(new Paragraph("www.mingribook.com"));
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
