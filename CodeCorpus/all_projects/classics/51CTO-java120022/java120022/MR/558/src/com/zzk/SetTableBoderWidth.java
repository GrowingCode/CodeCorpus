package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.lowagie.text.Document;
import com.lowagie.text.DocumentException;
import com.lowagie.text.Font;
import com.lowagie.text.Paragraph;
import com.lowagie.text.Table;
import com.lowagie.text.pdf.BaseFont;
import com.lowagie.text.pdf.PdfWriter;

public class SetTableBoderWidth {
    
    public static void main(String[] args) {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置表格的边框宽度.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont bfChinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基本字体
            Font fontChinese = new Font(bfChinese, 12, Font.NORMAL);// 实例化字体
            Table table1 = new Table(3);// 定义表格
            document.add(new Paragraph("                默认边框的表格", fontChinese));// 创建段落
            // 将单元格顺次的加入到表格，当一行充满时自动换行
            table1.addCell("Cell1.1");
            table1.addCell("Cell1.2");
            table1.addCell("Cell1.3");
            table1.addCell("Cell2.1");
            table1.addCell("Cell2.2");
            table1.addCell("Cell2.3");
            Table table2 = new Table(3);// 定义表格
            table2.setBorderWidth(3);// 设置表格的外边框宽度
            // 将单元格顺次的加入到表格，当一行充满时自动换行
            table2.addCell("Cell1.1");
            table2.addCell("Cell1.2");
            table2.addCell("Cell1.3");
            table2.addCell("Cell2.1");
            table2.addCell("Cell2.2");
            table2.addCell("Cell2.3");
            document.add(table1);// 将表格添加到文档
            document.add(new Paragraph("                改变外边框宽度为3的表格",
                    fontChinese));// 创建段落
            document.add(table2);// 将表格添加到文档
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
