package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class CreateTable {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\创建表格.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont Chinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
            Font FontChinese = new Font(Chinese, 12, Font.NORMAL);// 实例化字体
            PdfPTable table = new PdfPTable(3);// 创建表格
            // 将单元格顺次的加入到表格，当一行充满时自动换行
            table.addCell("1.1");
            table.addCell("1.2");
            table.addCell("1.3");
            table.addCell("2.1");
            table.addCell("2.2");
            table.addCell("2.3");
            table.addCell("3.1");
            table.addCell("3.2");
            table.addCell("3.3");
            document.add(new Paragraph("                    这是一个3行3列的表格\n.",
                    FontChinese));// 向文档添加内容
            document.add(table);// 将表格添加到文档
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
