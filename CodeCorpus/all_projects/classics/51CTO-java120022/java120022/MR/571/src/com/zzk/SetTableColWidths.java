package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class SetTableColWidths {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置表格的列宽.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont Chinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
            Font FontChinese = new Font(Chinese, 10, Font.NORMAL,
                    new BaseColor(90, 90, 90));// 实例化字体
            float[] widths = { 0.05f, 0.10f, 0.30f, 0.55f };// 设置列宽相对比例为
                                                            // 5%,10%,30%,55%
            PdfPTable table = new PdfPTable(widths);// 创建表格关联列宽
            table.addCell(new Paragraph("列宽为5%", FontChinese));// 将单元格顺次的加入到表格，当一行充满时自动换行
            table.addCell(new Paragraph("列宽为10%", FontChinese));
            table.addCell(new Paragraph("列宽为30%", FontChinese));
            table.addCell(new Paragraph("列宽为55%", FontChinese));
            table.addCell("w");
            table.addCell("1");
            table.addCell("2");
            table.addCell("3");
            table.addCell("h");
            table.addCell("1");
            table.addCell("2");
            table.addCell("3");
            table.addCell("s");
            table.addCell("1");
            table.addCell("2");
            table.addCell("3");
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
