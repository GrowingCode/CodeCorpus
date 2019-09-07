package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class SetTableWidth {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置表格宽度.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            float[] widths = { 0.05f, 0.10f, 0.30f, 0.55f };// 设置列宽相关比率为5%,10%,30%,55%
            PdfPTable table = new PdfPTable(widths);// 创建表格关联列宽
            table.setWidthPercentage(60);// 为表格设置百分比宽度
            // 将单元格顺次的加入到表格，当一行充满时自动换行
            table.addCell("5%");
            table.addCell("10%");
            table.addCell("30%");
            table.addCell("55%");
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
        }
    }
}
